# EMG Data Processing Pipeline

A comprehensive pipeline for processing EMG (Electromyography) data from Delsys Trigno sensors, detecting throws, calculating metrics, and storing results in a MySQL/HeidiSQL database.

## Features

- Process EMG data from Delsys Trigno sensors with FCU and FCR channels
- Detect throwing motions using adaptive thresholding on EMG signals
- Calculate comprehensive metrics for each throw:
  - Frequency domain metrics (median frequency, mean frequency, bandwidth)
  - Amplitude metrics (peak amplitude, RMS value)
  - Temporal metrics (rise time, contraction/relaxation times)
  - Workload metrics (throw integral, work rate)
- Store both raw time series data and throw-level metrics in a database
- Intuitive GUI for processing files and monitoring results
- Batch processing support for multiple files

## Setup

1. Clone the repo:
   ```
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. Create and activate a virtual environment (recommended):
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your database credentials:
   ```
   DB_HOST=your_host
   DB_USER=your_user
   DB_PASSWORD=your_password
   DB_NAME=your_db
   ```

5. Run the pipeline:
   ```
   python main.py
   ```

## Automation

- Use `run_pipeline.bat` and Windows Task Scheduler for daily automation.

## Database Setup

The pipeline automatically creates the necessary database tables:

- `emg_sessions`: Session-level information
- `emg_throws`: Throw-level metrics
- `emg_timeseries`: Raw time series data

Database credentials are stored in the code and can be modified in the GUI.

## Usage

### GUI Application

Run the GUI application for the easiest interaction:

```bash
python gui_app.py
```

The GUI provides:
- File/directory selection for processing
- Database connection testing
- Record count viewing
- Processing logs

### Command Line

Generate Report:
python emg_report_generator [sessionid]

Process a single file:
```bash
python main.py -f path/to/file.csv
```

Process a directory:
```bash
python main.py -d path/to/directory -r
```

Test database connection:
```bash
python main.py --test-db
```

## Data Format

The pipeline expects:
- File format: Delsys Trigno EMG data with FCU and FCR channels
- Filename format: MMDDYYYY_TraqID_Name_sessiontype.csv
  - Also supports MMDDYY date format

## Project Structure

- `gui_app.py`: Main GUI application
- `main.py`: Command line interface
- `emg_pipeline.py`: Core processing pipeline
- `emg_processor.py`: Signal processing and feature extraction
- `db_connector.py`: Database connection and operations
- 'emg_report_generator.py' : Report Generator

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.