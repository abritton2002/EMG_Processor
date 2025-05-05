import os
import sys
import re
from collections import defaultdict
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded .env file for environment variables")
except ImportError:
    print("Warning: dotenv package not installed. Using environment variables directly.")
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")

def summarize_log(log_path):
    summary_lines = []
    processed_files = []
    successful_files = []
    errors = []
    logs = []
    
    try:
        with open(log_path, "r", encoding="utf-8", errors='replace') as f:
            logs = f.readlines()
    except Exception as e:
        return f"Error reading log file: {e}"
    
    # Extract pipeline info and process files
    for line in logs:
        # Add processed file info
        if "Processing file:" in line:
            file_path = line.split("Processing file:")[1].strip()
            base_name = os.path.basename(file_path)
            processed_files.append(base_name)
        
        # Add successful saves
        elif "Successfully saved" in line and "to database" in line:
            saved_file = line.split("Successfully saved")[1].split("to database")[0].strip()
            successful_files.append(saved_file)
        
        # Collect errors
        elif "Error:" in line or "ERROR:" in line or "Failed to" in line:
            errors.append(line.strip())
    
    # Extract summary from final lines
    success_summary = None
    for line in logs:
        if "Successfully processed" in line and "files" in line:
            success_summary = line.strip()
            break
    
    # Build the summary
    summary_lines.append("*EMG Pipeline Summary*")
    
    if processed_files:
        summary_lines.append(f"\nProcessed {len(processed_files)} files:")
        for file in processed_files:
            # Extract just the base filename (without path) for comparison
            file_base = os.path.basename(file) if os.path.sep in file else file
            # Check if any of the successful files matches or contains this filename
            is_successful = False
            for successful in successful_files:
                if successful == file_base or successful in file_base or file_base in successful:
                    is_successful = True
                    break
            
            status = "[SUCCESS]" if is_successful else "[FAILED]"
            summary_lines.append(f"- {file}: {status}")
    
    if errors:
        summary_lines.append("\n*Errors/Warnings:*")
        # Limit to important errors
        for i, error in enumerate(errors[:10]):
            summary_lines.append(f"- {error}")
        if len(errors) > 10:
            summary_lines.append(f"...and {len(errors) - 10} more errors")
    
    if success_summary:
        summary_lines.append(f"\n*Final Result:* {success_summary}")
    
    if not summary_lines:
        return "No information could be extracted from the log file."
    
    return "\n".join(summary_lines)

def send_to_slack(message):
    # Get Slack credentials
    SLACK_TOKEN = os.getenv("SLACK_BOT_TOKEN")
    CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID")
    
    # Debug information
    print(f"Slack token available: {'Yes' if SLACK_TOKEN else 'No'}")
    print(f"Slack channel available: {'Yes' if CHANNEL_ID else 'No'}")
    
    if SLACK_TOKEN:
        # Mask the token when printing for security
        visible_part = SLACK_TOKEN[:10] if len(SLACK_TOKEN) > 10 else SLACK_TOKEN[:5]
        print(f"Token starts with: {visible_part}...")
    
    # Check if credentials are available
    if not SLACK_TOKEN:
        print("ERROR: SLACK_BOT_TOKEN environment variable is not set or empty.")
        print("Please set this variable in your environment or .env file.")
        print("The summary was not sent to Slack.")
        return False
    
    if not CHANNEL_ID:
        print("ERROR: SLACK_CHANNEL_ID environment variable is not set or empty.")
        print("Please set this variable in your environment or .env file.")
        print("The summary was not sent to Slack.")
        return False
    
    # Attempt to send message
    try:
        print(f"Attempting to send message to Slack channel: {CHANNEL_ID}")
        client = WebClient(token=SLACK_TOKEN)
        response = client.chat_postMessage(
            channel=CHANNEL_ID,
            text=message
        )
        print(f"Message successfully sent to Slack channel: {CHANNEL_ID}")
        return True
    except SlackApiError as e:
        error = e.response.get("error", "Unknown error")
        print(f"ERROR sending summary to Slack: {error}")
        
        # Provide more helpful error messages
        if error == "not_authed":
            print("The Slack token is invalid or has been revoked.")
            print("Please check your SLACK_BOT_TOKEN environment variable.")
        elif error == "channel_not_found":
            print(f"The channel ID '{CHANNEL_ID}' was not found.")
            print("Please check your SLACK_CHANNEL_ID environment variable.")
        elif error == "invalid_auth":
            print("Authentication failed. Your token may have expired.")
        elif error == "account_inactive":
            print("The Slack account associated with your token is inactive.")
        
        print("\nTo fix this issue:")
        print("1. Make sure your Slack bot token is correct and still valid")
        print("2. Ensure the bot has been added to the channel you're trying to post to")
        print("3. Check that the channel ID is correct")
        print("4. Verify the bot has the 'chat:write' permission scope")
        
        return False
    except Exception as e:
        print(f"Unexpected error when sending to Slack: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python summarize_and_send_log.py <logfile>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    
    # Check if log file exists
    if not os.path.exists(log_file):
        print(f"ERROR: Log file not found: {log_file}")
        sys.exit(1)
    
    # Generate summary
    print(f"Generating summary from log file: {log_file}")
    summary = summarize_log(log_file)
    
    # Print summary to console
    print("\n===== SUMMARY =====")
    print(summary)
    print("===================\n")
    
    # Save the summary to a text file regardless of Slack status
    summary_file = f"{os.path.splitext(log_file)[0]}_summary.txt"
    try:
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"Summary saved to file: {summary_file}")
    except Exception as e:
        print(f"Error saving summary to file: {e}")
    
    # Send to Slack
    print("Attempting to send summary to Slack...")
    success = send_to_slack(summary)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)