import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import threading
import os
import sys
import logging
import datetime
from pathlib import Path
from dotenv import load_dotenv
from tkcalendar import Calendar, DateEntry
# Import our pipeline modules
from emg_pipeline import EMGPipeline
from db_connector import DBConnector
from emg_report_generator import EMGProfessionalReport

# Load environment variables
load_dotenv()

# Import our pipeline modules
from emg_pipeline import EMGPipeline
from db_connector import DBConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('emg_pipeline.log')
    ]
)

logger = logging.getLogger(__name__)

class EMGPipelineGUI:
    """GUI for the EMG Data Processing Pipeline."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("EMG Data Processing Pipeline")
        self.root.geometry("800x600")
        self.root.minsize(700, 500)
        
        # Database configuration from environment variables
        self.db_config = {
            'host': os.getenv('DB_HOST', ''),
            'user': os.getenv('DB_USER', ''),
            'password': os.getenv('DB_PASSWORD', ''),
            'database': os.getenv('DB_NAME', '')
        }
        
        # Initialize pipeline
        self.pipeline = EMGPipeline(
            db_config=self.db_config,
            batch_size=int(os.getenv('BATCH_SIZE', '1000'))
        )
        
        # Create the GUI elements
        self.create_widgets()
        
        # Check environment setup
        self.check_env_setup()
        
        # Update status
        self.update_status("Ready")
    
    def check_env_setup(self):
        """Check if environment variables are set and warn if not."""
        missing_vars = []
        for key, value in self.db_config.items():
            if not value:
                missing_vars.append(f"DB_{key.upper()}")
        
        if missing_vars:
            messagebox.showwarning(
                "Environment Setup", 
                f"The following environment variables are not set: {', '.join(missing_vars)}\n\n"
                "Please configure database settings in the Database tab and save to .env file."
            )
    
    def create_widgets(self):
        """Create all the GUI widgets."""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create title
        title_label = ttk.Label(
            main_frame, 
            text="EMG Data Processing Pipeline", 
            font=("Helvetica", 16, "bold")
        )
        title_label.pack(pady=(0, 10))
        
        # Create tabbed interface
        tab_control = ttk.Notebook(main_frame)
        
        # File Processing Tab
        process_tab = ttk.Frame(tab_control)
        tab_control.add(process_tab, text="Process Files")
        self.setup_process_tab(process_tab)
        
        # Database Tab
        db_tab = ttk.Frame(tab_control)
        tab_control.add(db_tab, text="Database")
        self.setup_db_tab(db_tab)
        
        # Reports Tab
        reports_tab = ttk.Frame(tab_control)
        tab_control.add(reports_tab, text="Generate Reports")
        self.setup_report_tab(reports_tab)
        
        # Help Tab
        help_tab = ttk.Frame(tab_control)
        tab_control.add(help_tab, text="Help")
        self.setup_help_tab(help_tab)
        
        tab_control.pack(expand=1, fill=tk.BOTH)
        
        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(status_frame, text="Status:").pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT, padx=(5, 0))
    
    def setup_process_tab(self, tab):
        """Set up the file processing tab."""
        # File selection frame
        file_frame = ttk.LabelFrame(tab, text="Select EMG Data", padding="10")
        file_frame.pack(fill=tk.X, pady=10)
        
        # Radio buttons for file or directory
        self.process_type = tk.StringVar(value="file")
        ttk.Radiobutton(
            file_frame, 
            text="Process Single File", 
            variable=self.process_type, 
            value="file",
            command=self.update_file_selection
        ).grid(row=0, column=0, sticky=tk.W)
        
        ttk.Radiobutton(
            file_frame, 
            text="Process Directory", 
            variable=self.process_type, 
            value="directory",
            command=self.update_file_selection
        ).grid(row=0, column=1, sticky=tk.W)
        
        # File/directory path
        ttk.Label(file_frame, text="Path:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.path_var = tk.StringVar()
        path_entry = ttk.Entry(file_frame, textvariable=self.path_var, width=50)
        path_entry.grid(row=1, column=1, sticky=tk.EW, padx=(5, 5), pady=(10, 0))
        
        self.browse_button = ttk.Button(
            file_frame, 
            text="Browse...", 
            command=self.browse_path
        )
        self.browse_button.grid(row=1, column=2, padx=(0, 5), pady=(10, 0))
        
        # Recursive option
        self.recursive_var = tk.BooleanVar(value=False)
        self.recursive_check = ttk.Checkbutton(
            file_frame, 
            text="Process Recursively (for directories)", 
            variable=self.recursive_var
        )
        self.recursive_check.grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))
        
        # Dry run option
        self.dry_run_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            file_frame, 
            text="Dry Run (don't save to database)", 
            variable=self.dry_run_var
        ).grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))
        
        # Process button
        process_button = ttk.Button(
            file_frame, 
            text="Start Processing", 
            command=self.start_processing
        )
        process_button.grid(row=4, column=0, columnspan=3, pady=(10, 5))
        
        # Output frame
        output_frame = ttk.LabelFrame(tab, text="Processing Log", padding="10")
        output_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Log text widget
        self.log_text = tk.Text(output_frame, wrap=tk.WORD, height=15)
        self.log_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(output_frame, command=self.log_text.yview)
        scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        # Configure grid
        file_frame.columnconfigure(1, weight=1)
        
        # Update initial state
        self.update_file_selection()
    
    def setup_db_tab(self, tab):
        """Set up the database tab."""
        # Database connection frame
        db_frame = ttk.LabelFrame(tab, text="Database Connection", padding="10")
        db_frame.pack(fill=tk.X, pady=10)
        
        # Host
        ttk.Label(db_frame, text="Host:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.host_var = tk.StringVar(value=self.db_config['host'])
        ttk.Entry(db_frame, textvariable=self.host_var, width=30).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # User
        ttk.Label(db_frame, text="User:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.user_var = tk.StringVar(value=self.db_config['user'])
        ttk.Entry(db_frame, textvariable=self.user_var, width=30).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Password
        ttk.Label(db_frame, text="Password:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.password_var = tk.StringVar(value=self.db_config['password'])
        ttk.Entry(db_frame, textvariable=self.password_var, width=30, show="*").grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Database
        ttk.Label(db_frame, text="Database:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.database_var = tk.StringVar(value=self.db_config['database'])
        ttk.Entry(db_frame, textvariable=self.database_var, width=30).grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Test connection button
        ttk.Button(
            db_frame, 
            text="Test Connection", 
            command=self.test_db_connection
        ).grid(row=4, column=0, pady=(10, 5))
        
        # Save config button
        ttk.Button(
            db_frame, 
            text="Update Connection", 
            command=self.update_db_config
        ).grid(row=4, column=1, pady=(10, 5))
        
        # Save to .env button
        ttk.Button(
            db_frame, 
            text="Save to .env File", 
            command=self.save_to_env
        ).grid(row=5, column=0, columnspan=2, pady=(5, 5))
        
        # Database status
        ttk.Label(db_frame, text="Status:").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.db_status_var = tk.StringVar(value="Not tested")
        ttk.Label(db_frame, textvariable=self.db_status_var).grid(row=6, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Environment variables status
        env_frame = ttk.LabelFrame(tab, text="Environment Status", padding="10")
        env_frame.pack(fill=tk.X, pady=10)
        
        self.env_status_text = tk.Text(env_frame, wrap=tk.WORD, height=4)
        self.env_status_text.pack(fill=tk.BOTH, expand=True)
        self.update_env_status()
        
        # Database operations frame
        ops_frame = ttk.LabelFrame(tab, text="Database Operations", padding="10")
        ops_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Get table counts button
        ttk.Button(
            ops_frame, 
            text="Get Record Counts", 
            command=self.get_record_counts
        ).pack(pady=5)
        
        # Results text
        self.db_results_text = tk.Text(ops_frame, wrap=tk.WORD, height=10)
        self.db_results_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        # Scrollbar
        db_scrollbar = ttk.Scrollbar(ops_frame, command=self.db_results_text.yview)
        db_scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
        self.db_results_text.config(yscrollcommand=db_scrollbar.set)
    
    def setup_report_tab(self, tab):
        """Set up the report generation tab."""
        # Report generation frame
        report_frame = ttk.LabelFrame(tab, text="Generate EMG Reports", padding="10")
        report_frame.pack(fill=tk.X, pady=10)
        
        # Report source selection
        self.report_source_type = tk.StringVar(value="date")
        ttk.Radiobutton(
            report_frame, 
            text="By Date", 
            variable=self.report_source_type, 
            value="date",
            command=self.update_report_source_ui
        ).grid(row=0, column=0, sticky=tk.W, pady=(5, 0))
        
        ttk.Radiobutton(
            report_frame, 
            text="By Session ID", 
            variable=self.report_source_type, 
            value="session_id",
            command=self.update_report_source_ui
        ).grid(row=0, column=1, sticky=tk.W, pady=(5, 0))
        
        # Date selection
        ttk.Label(report_frame, text="Date:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.report_date_var = tk.StringVar()
        date_entry = ttk.Entry(report_frame, textvariable=self.report_date_var, width=20)
        date_entry.grid(row=1, column=1, sticky=tk.W, padx=(5, 5), pady=(10, 0))
        
        # Date picker button
        date_button = ttk.Button(
            report_frame, 
            text="Choose Date", 
            command=self.open_date_picker
        )
        date_button.grid(row=1, column=2, padx=(0, 5), pady=(10, 0))
        
        # Sessions list (for date-based selection)
        self.sessions_frame = ttk.LabelFrame(report_frame, text="Sessions", padding="10")
        self.sessions_frame.grid(row=2, column=0, columnspan=3, sticky='ew', pady=(10, 0))
        
        # Sessions list scrollable
        sessions_scroll = ttk.Scrollbar(self.sessions_frame)
        sessions_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.sessions_list = tk.Listbox(
            self.sessions_frame, 
            selectmode=tk.MULTIPLE, 
            yscrollcommand=sessions_scroll.set,
            height=5
        )
        self.sessions_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sessions_scroll.config(command=self.sessions_list.yview)
        
        # Find sessions button
        find_sessions_button = ttk.Button(
            report_frame, 
            text="Find Sessions", 
            command=self.find_sessions
        )
        find_sessions_button.grid(row=3, column=0, columnspan=3, pady=(10, 0))
        
        # Output directory selection
        ttk.Label(report_frame, text="Output Directory:").grid(row=4, column=0, sticky=tk.W, pady=(10, 0))
        self.report_output_var = tk.StringVar(value="reports")
        output_entry = ttk.Entry(report_frame, textvariable=self.report_output_var, width=50)
        output_entry.grid(row=4, column=1, sticky=tk.EW, padx=(5, 5), pady=(10, 0))
        
        # Browse output directory button
        ttk.Button(
            report_frame, 
            text="Browse...", 
            command=self.browse_report_output_dir
        ).grid(row=4, column=2, padx=(0, 5), pady=(10, 0))
        
        # Generate report button
        generate_button = ttk.Button(
            report_frame, 
            text="Generate Reports", 
            command=self.start_report_generation
        )
        generate_button.grid(row=5, column=0, columnspan=3, pady=(10, 5))
        
        # Output frame for logs
        output_frame = ttk.LabelFrame(tab, text="Report Generation Log", padding="10")
        output_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Log text widget
        self.report_log_text = tk.Text(output_frame, wrap=tk.WORD, height=15)
        self.report_log_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(output_frame, command=self.report_log_text.yview)
        scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
        self.report_log_text.config(yscrollcommand=scrollbar.set)
        
        # Configure grid
        report_frame.columnconfigure(1, weight=1)

    def update_report_source_ui(self):
        """Update UI based on report source type."""
        source_type = self.report_source_type.get()
        
        if source_type == "date":
            # Show date-related widgets
            self.report_date_var.set("")
            self.sessions_list.delete(0, tk.END)
            self.sessions_frame.grid()
        else:
            # Hide date and sessions widgets
            self.sessions_frame.grid_remove()
            self.report_date_var.set("")

    def browse_report_output_dir(self):
        """Browse for report output directory."""
        dir_path = filedialog.askdirectory(title="Select Output Directory for Reports")
        if dir_path:
            self.report_output_var.set(dir_path)

    def open_date_picker(self):
        """Open a date picker dialog."""
        from tkcalendar import Calendar, DateEntry
        
        # Create a top-level window for date selection
        top = tk.Toplevel(self.root)
        top.title("Select Date")
        
        # Create a calendar widget
        cal = DateEntry(
            top, 
            width=12, 
            background='darkblue', 
            foreground='white', 
            borderwidth=2,
            date_pattern='yyyy-mm-dd'
        )
        cal.pack(padx=10, pady=10)
        
        def set_date():
            """Set the selected date and close the window."""
            selected_date = cal.get_date().strftime('%Y-%m-%d')
            self.report_date_var.set(selected_date)
            top.destroy()
        
        # Add a button to confirm date selection
        ttk.Button(top, text="Select", command=set_date).pack(pady=10)

    def find_sessions(self):
        """Find sessions for the selected date."""
        # Clear previous sessions
        self.sessions_list.delete(0, tk.END)
        
        # Get selected date
        selected_date = self.report_date_var.get()
        
        if not selected_date:
            messagebox.showerror("Error", "Please select a date first.")
            return
        
        # Connect to database and fetch sessions
        try:
            db = DBConnector(self.db_config)
            conn = db.connect()
            
            if not conn:
                messagebox.showerror("Error", "Failed to connect to database.")
                return
            
            cursor = conn.cursor()
            
            # Query to find sessions for the selected date
            cursor.execute("""
                SELECT numeric_id, athlete_name, session_type, collection_date 
                FROM emg_sessions 
                WHERE DATE(date_recorded) = %s OR DATE(collection_date) = %s
            """, (selected_date, selected_date))
            
            sessions = cursor.fetchall()
            
            if not sessions:
                messagebox.showinfo("No Sessions", f"No sessions found for {selected_date}")
                return
            
            # Populate sessions list
            for session in sessions:
                numeric_id, athlete_name, session_type, collection_date = session
                display_text = f"ID: {numeric_id} | Athlete: {athlete_name} | Type: {session_type}"
                self.sessions_list.insert(tk.END, display_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to retrieve sessions: {str(e)}")
        finally:
            # Always close the connection
            if 'conn' in locals():
                db.disconnect()

    def start_report_generation(self):
        """Start generating reports."""
        # Clear previous log
        self.report_log_text.delete(1.0, tk.END)
        
        # Get output directory
        output_dir = self.report_output_var.get()
        
        # Validate output directory
        if not output_dir:
            messagebox.showerror("Error", "Please specify an output directory.")
            return
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create report generator
        report_generator = EMGProfessionalReport(
            db_config=self.db_config, 
            output_dir=output_dir
        )
        
        # Determine sessions to process
        if self.report_source_type.get() == "session_id":
            # Single session ID
            session_ids = [self.report_session_var.get()]
        else:
            # Multiple sessions from list
            selected_indices = self.sessions_list.curselection()
            
            if not selected_indices:
                messagebox.showerror("Error", "Please select at least one session.")
                return
            
                # Extract session IDs from the listbox selection
    # Extract session IDs from the listbox selection
            session_ids = []
            for index in selected_indices:
                session_text = self.sessions_list.get(index)
                # Extract numeric_id from the display text
                numeric_id = session_text.split('|')[0].split(':')[1].strip()
                session_ids.append(numeric_id)
        # Run report generation in a thread
        def generate_reports():
            success_count = 0
            for session_id in session_ids:
                try:
                    self.append_report_log(f"Generating report for session: {session_id}")
                    report_path = report_generator.generate_report(session_id)
                    
                    if report_path:
                        success_count += 1
                        self.append_report_log(f"  ✓ Report generated: {report_path}")
                    else:
                        self.append_report_log(f"  ✗ Failed to generate report for session {session_id}")
                
                except Exception as e:
                    self.append_report_log(f"Error generating report for session {session_id}: {str(e)}")
            
            # Final summary
            self.append_report_log(f"Report generation complete. "
                                    f"Successfully generated {success_count}/{len(session_ids)} reports.")
        
        # Start report generation in a separate thread
        threading.Thread(target=generate_reports, daemon=True).start()

    def append_report_log(self, message):
        """Append message to report generation log."""
        def log_update():
            self.report_log_text.insert(tk.END, message + "\n")
            self.report_log_text.see(tk.END)
        
        # Use after method to ensure thread-safe GUI update
        self.root.after(0, log_update)

    def update_env_status(self):
        """Update the environment variables status display."""
        self.env_status_text.config(state=tk.NORMAL)
        self.env_status_text.delete(1.0, tk.END)
        
        # Check which environment variables are set
        status = {
            'DB_HOST': os.getenv('DB_HOST') is not None,
            'DB_USER': os.getenv('DB_USER') is not None,
            'DB_PASSWORD': os.getenv('DB_PASSWORD') is not None,
            'DB_NAME': os.getenv('DB_NAME') is not None,
            'BATCH_SIZE': os.getenv('BATCH_SIZE') is not None
        }
        
        # Display status
        self.env_status_text.insert(tk.END, "Environment Variables Status:\n")
        for var, is_set in status.items():
            status_text = "✓ Set" if is_set else "✗ Not Set"
            self.env_status_text.insert(tk.END, f"{var}: {status_text}\n")
        
        self.env_status_text.config(state=tk.DISABLED)
    
    def save_to_env(self):
        """Save current database configuration to .env file."""
        try:
            # Read existing .env file if it exists to preserve other variables
            env_vars = {}
            if os.path.exists('.env'):
                with open('.env', 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            env_vars[key.strip()] = value.strip()
            
            # Update database variables
            env_vars['DB_HOST'] = self.host_var.get()
            env_vars['DB_USER'] = self.user_var.get()
            env_vars['DB_PASSWORD'] = self.password_var.get()
            env_vars['DB_NAME'] = self.database_var.get()
            env_vars['BATCH_SIZE'] = env_vars.get('BATCH_SIZE', '1000')
            
            # Write to .env file
            with open('.env', 'w') as f:
                f.write("# Database Configuration\n")
                f.write(f"DB_HOST={env_vars['DB_HOST']}\n")
                f.write(f"DB_USER={env_vars['DB_USER']}\n")
                f.write(f"DB_PASSWORD={env_vars['DB_PASSWORD']}\n")
                f.write(f"DB_NAME={env_vars['DB_NAME']}\n\n")
                f.write("# Processing Configuration\n")
                f.write(f"BATCH_SIZE={env_vars['BATCH_SIZE']}\n\n")
                
                # Write other variables
                if 'DEFAULT_DATA_DIR' in env_vars:
                    f.write("# File paths for default locations\n")
                    f.write(f"DEFAULT_DATA_DIR={env_vars['DEFAULT_DATA_DIR']}\n")
            
            messagebox.showinfo("Success", "Database configuration saved to .env file!")
            
            # Reload environment variables
            load_dotenv(override=True)
            
            # Update the environment status display
            self.update_env_status()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save .env file: {str(e)}")
    
    def setup_help_tab(self, tab):
        """Set up the help tab."""
        # Help text
        help_text = """
EMG Data Processing Pipeline Help

This application processes EMG data files and stores the results in a database.

File Processing:
- Process a single EMG file or a directory of files
- Check "Process Recursively" to include subdirectories
- Use "Dry Run" to process files without saving to the database

Database:
- Configure your database connection
- Test the connection before processing
- View record counts in the database
- Save connection settings to .env file for future use

File Format:
- The application expects files in Delsys Trigno format
- Filename format should be: MMDDYYYY_TraqID_Name_sessiontype.csv
- The file should contain FCU and FCR channel data

Processing:
- The pipeline detects throws based on EMG signal patterns
- Calculates metrics for each throw (frequency, amplitude, etc.)
- Stores both raw time series data and throw-level metrics
- Each file is processed independently

Environment Variables:
- The application uses a .env file for configuration
- You can modify database settings in the Database tab
- Click "Save to .env File" to update the configuration

First-Time Setup:
1. Go to the Database tab
2. Enter your database connection details
3. Click "Save to .env File"
4. Test the connection

For more details, see the README file or check the log file (emg_pipeline.log).
        """
        
        # Text widget for help
        help_text_widget = tk.Text(tab, wrap=tk.WORD, padx=10, pady=10)
        help_text_widget.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        help_text_widget.insert(tk.END, help_text)
        help_text_widget.config(state=tk.DISABLED)
        
        # Scrollbar
        help_scrollbar = ttk.Scrollbar(tab, command=help_text_widget.yview)
        help_scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
        help_text_widget.config(yscrollcommand=help_scrollbar.set)
    
    def update_file_selection(self):
        """Update UI based on file selection type."""
        if self.process_type.get() == "file":
            self.recursive_check.config(state=tk.DISABLED)
            self.browse_button.config(command=self.browse_file)
        else:
            self.recursive_check.config(state=tk.NORMAL)
            self.browse_button.config(command=self.browse_directory)
    
    def browse_file(self):
        """Open file browser dialog."""
        file_path = filedialog.askopenfilename(
            title="Select EMG Data File",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.path_var.set(file_path)
    
    def browse_directory(self):
        """Open directory browser dialog."""
        dir_path = filedialog.askdirectory(title="Select Directory")
        if dir_path:
            self.path_var.set(dir_path)
    
    def browse_path(self):
        """Browse for file or directory based on selection."""
        if self.process_type.get() == "file":
            self.browse_file()
        else:
            self.browse_directory()
    
    def update_status(self, message):
        """Update status bar."""
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def append_log(self, message):
        """Append message to log text widget."""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def clear_log(self):
        """Clear log text widget."""
        self.log_text.delete(1.0, tk.END)
    
    def start_processing(self):
        """Start processing EMG data files."""
        # Check if environment variables are set
        missing_vars = []
        for key, value in self.db_config.items():
            if not value and not self.dry_run_var.get():
                missing_vars.append(f"DB_{key.upper()}")
        
        if missing_vars and not self.dry_run_var.get():
            messagebox.showerror(
                "Configuration Error", 
                f"The following environment variables are not set: {', '.join(missing_vars)}\n\n"
                "Please configure database settings in the Database tab and save to .env file."
            )
            return
        
        path = self.path_var.get()
        if not path:
            messagebox.showerror("Error", "Please select a file or directory.")
            return
        
        if not os.path.exists(path):
            messagebox.showerror("Error", "The selected path does not exist.")
            return
        
        # Update UI
        self.clear_log()
        self.append_log(f"Starting processing of: {path}")
        self.update_status("Processing...")
        
        # Create updated pipeline with current settings
        self.update_db_config()
        
        # Process in a separate thread to keep UI responsive
        processing_thread = threading.Thread(
            target=self.run_processing,
            args=(path,)
        )
        processing_thread.daemon = True
        processing_thread.start()
    
    def run_processing(self, path):
        """Run the processing in a separate thread."""
        try:
            is_file = self.process_type.get() == "file"
            recursive = self.recursive_var.get()
            dry_run = self.dry_run_var.get()
            
            self.append_log(f"{'Dry run' if dry_run else 'Processing'} started at {datetime.datetime.now().strftime('%H:%M:%S')}")
            
            if is_file:
                # Process single file
                self.append_log(f"Processing file: {os.path.basename(path)}")
                
                if dry_run:
                    # Just process without saving to database
                    processed_data = self.pipeline.process_file(path)
                    if processed_data:
                        session_id = processed_data['session_data']['session_id']
                        throws_count = len(processed_data['throw_data'])
                        timeseries_rows = len(processed_data['timeseries_data'])
                        self.append_log(f"File processed successfully: {session_id}")
                        self.append_log(f"Detected {throws_count} throws")
                        self.append_log(f"Processed {timeseries_rows} time points")
                        self.update_status("Processing complete (dry run)")
                    else:
                        self.append_log(f"Failed to process file: {path}")
                        self.update_status("Processing failed")
                else:
                    # Process and save to database
                    success = self.pipeline.run_single_file(path)
                    if success:
                        self.append_log(f"File processed and saved successfully")
                        self.update_status("Processing complete")
                    else:
                        self.append_log(f"Failed to process or save file")
                        self.update_status("Processing failed")
            else:
                # Process directory
                self.append_log(f"Processing directory: {path}")
                self.append_log(f"Recursive: {recursive}")
                
                # Count files before processing
                file_count = 0
                if recursive:
                    for root, _, files in os.walk(path):
                        file_count += sum(1 for f in files if f.endswith(('.csv', '.txt')))
                else:
                    file_count = sum(1 for f in os.listdir(path) 
                                  if os.path.isfile(os.path.join(path, f)) and 
                                  f.endswith(('.csv', '.txt')))
                
                self.append_log(f"Found {file_count} files to process")
                
                if file_count == 0:
                    self.append_log("No EMG data files found")
                    self.update_status("No files found")
                    return
                
                # Process directory
                if dry_run:
                    self.append_log("Dry run mode: files will be processed but not saved to database")
                
                summary = self.pipeline.process_directory(path, recursive)
                
                self.append_log(f"Directory processing complete")
                self.append_log(f"Successfully processed {summary['processed']} of {summary['total']} files")
                if summary['failed'] > 0:
                    self.append_log(f"Failed to process {summary['failed']} files")
                
                if summary['success']:
                    self.update_status(f"Processed {summary['processed']}/{summary['total']} files")
                else:
                    self.update_status("Processing completed with errors")
            
            self.append_log(f"Processing finished at {datetime.datetime.now().strftime('%H:%M:%S')}")
            
        except Exception as e:
            self.append_log(f"Error during processing: {str(e)}")
            self.update_status("Error during processing")
            logger.exception("Error in processing thread")
    
    def test_db_connection(self):
        """Test the database connection."""
        self.update_db_config()
        
        # Check if all required fields are provided
        if not all([self.db_config['host'], self.db_config['user'], 
                    self.db_config['password'], self.db_config['database']]):
            messagebox.showerror("Error", "Please fill in all database connection fields")
            return
        
        db = DBConnector(self.db_config)
        if db.test_connection():
            self.db_status_var.set("Connected")
            messagebox.showinfo("Success", "Database connection successful!")
        else:
            self.db_status_var.set("Failed")
            messagebox.showerror("Error", "Database connection failed. Check logs for details.")
    
    def update_db_config(self):
        """Update database configuration from UI inputs."""
        self.db_config = {
            'host': self.host_var.get(),
            'user': self.user_var.get(),
            'password': self.password_var.get(),
            'database': self.database_var.get()
        }
        
        # Update pipeline with new config
        self.pipeline = EMGPipeline(
            db_config=self.db_config,
            batch_size=int(os.getenv('BATCH_SIZE', '1000'))
        )
        
        self.db_status_var.set("Updated")
    
    def get_record_counts(self):
        """Get record counts from the database tables."""
        self.update_db_config()
        
        # Check if database configuration is complete
        if not all([self.db_config['host'], self.db_config['user'], 
                    self.db_config['password'], self.db_config['database']]):
            self.db_results_text.delete(1.0, tk.END)
            self.db_results_text.insert(tk.END, "Error: Database configuration incomplete.\n\n"
                                      "Please fill in all database connection fields and save to .env file.")
            return
        
        db = DBConnector(self.db_config)
        conn = db.connect()
        
        if not conn:
            self.db_results_text.delete(1.0, tk.END)
            self.db_results_text.insert(tk.END, "Failed to connect to database.")
            return
        
        try:
            cursor = conn.cursor()
            
            # Clear results
            self.db_results_text.delete(1.0, tk.END)
            
            # Check if tables exist
            cursor.execute("SHOW TABLES LIKE 'emg_%'")
            tables = cursor.fetchall()
            
            if not tables:
                self.db_results_text.insert(tk.END, "No EMG tables found in the database.")
                db.disconnect()
                return
            
            # Get counts for each table
            self.db_results_text.insert(tk.END, "Database Record Counts:\n\n")
            
            # Sessions count
            cursor.execute("SELECT COUNT(*) FROM emg_sessions")
            sessions_count = cursor.fetchone()[0]
            self.db_results_text.insert(tk.END, f"EMG Sessions: {sessions_count}\n")
            
            # Throws count
            cursor.execute("SELECT COUNT(*) FROM emg_throws")
            throws_count = cursor.fetchone()[0]
            self.db_results_text.insert(tk.END, f"EMG Throws: {throws_count}\n")
            
            # Time series count
            cursor.execute("SELECT COUNT(*) FROM emg_timeseries")
            timeseries_count = cursor.fetchone()[0]
            self.db_results_text.insert(tk.END, f"EMG Time Series Points: {timeseries_count}\n\n")
            
            # Get recent sessions
            self.db_results_text.insert(tk.END, "Recent Sessions:\n")
            cursor.execute("""
            SELECT session_id, date_recorded, athlete_name, session_type, collection_date 
            FROM emg_sessions 
            ORDER BY collection_date DESC 
            LIMIT 5
            """)
            
            recent_sessions = cursor.fetchall()
            if recent_sessions:
                for session in recent_sessions:
                    session_id, date_recorded, athlete_name, session_type, collection_date = session
                    self.db_results_text.insert(tk.END, f"- {session_id}\n")
                    self.db_results_text.insert(tk.END, f"  Date: {date_recorded}, Athlete: {athlete_name}\n")
                    self.db_results_text.insert(tk.END, f"  Type: {session_type}, Processed: {collection_date}\n\n")
            else:
                self.db_results_text.insert(tk.END, "No sessions found.\n")
            
        except Exception as e:
            self.db_results_text.delete(1.0, tk.END)
            self.db_results_text.insert(tk.END, f"Error getting record counts: {str(e)}")
            logger.exception("Error getting record counts")
        finally:
            db.disconnect()

def main():
    """Main entry point for the GUI application."""
    root = tk.Tk()
    app = EMGPipelineGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()