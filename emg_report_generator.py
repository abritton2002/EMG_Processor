import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.colors import HexColor
from io import BytesIO
import logging
import datetime
import pymysql
from db_connector import DBConnector
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

class EMGReportGenerator:
    """Generates PDF reports for EMG muscle activity data."""
    
    def __init__(self, db_config=None, output_dir="reports"):
        self.db = DBConnector(db_config)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._setup_styles()
        logger.info("EMG Report Generator initialized")

    def _setup_styles(self):
        self.styles.add(ParagraphStyle(name='CoverTitle', fontName='Helvetica-Bold', fontSize=24, textColor=HexColor('#1E3A8A'), alignment=TA_CENTER, spaceAfter=24))
        self.styles.add(ParagraphStyle(name='CoverBody', fontName='Helvetica', fontSize=14, textColor=HexColor('#4B5563'), alignment=TA_CENTER, spaceAfter=12))
        self.styles.add(ParagraphStyle(name='SectionTitle', fontName='Helvetica-Bold', fontSize=14, textColor=HexColor('#1E3A8A'), alignment=TA_LEFT, spaceAfter=8))
        self.styles.add(ParagraphStyle(name='SectionBody', fontName='Helvetica', fontSize=10, textColor=HexColor('#4B5563'), alignment=TA_LEFT, spaceAfter=6))

    def _resolve_session_id(self, session_id):
        if isinstance(session_id, int) or (isinstance(session_id, str) and session_id.isdigit()):
            return session_id
        with self.db.connect() as conn:
            if not conn:
                logger.error("Failed to connect to database")
                return None
            cursor = conn.cursor()
            cursor.execute("SELECT numeric_id FROM emg_sessions WHERE filename = %s", (session_id,))
            result = cursor.fetchone()
            if not result:
                logger.error(f"Session {session_id} not found")
                return None
            return result[0]

    def get_session_info(self, session_id):
        with self.db.connect() as conn:
            if not conn:
                logger.error("Failed to connect to database")
                return {}
            try:
                cursor = conn.cursor(pymysql.cursors.DictCursor)
                query = "SELECT * FROM emg_sessions WHERE numeric_id = %s" if \
                        (isinstance(session_id, int) or (isinstance(session_id, str) and session_id.isdigit())) \
                        else "SELECT * FROM emg_sessions WHERE filename = %s"
                cursor.execute(query, (session_id,))
                return cursor.fetchone() or {}
            except Exception as e:
                logger.error(f"Error retrieving session info for {session_id}: {e}")
                return {}

    def get_throws_for_session(self, session_id):
        numeric_id = self._resolve_session_id(session_id)
        if numeric_id is None:
            return pd.DataFrame()
        with self.db.connect() as conn:
            if not conn:
                logger.error("Failed to connect to database")
                return pd.DataFrame()
            try:
                query = "SELECT * FROM emg_throws WHERE session_numeric_id = %s ORDER BY trial_number"
                cursor = conn.cursor(pymysql.cursors.DictCursor)
                cursor.execute(query, (numeric_id,))
                throws = cursor.fetchall()
                df = pd.DataFrame(throws)
                if not df.empty:
                    df.rename(columns={'trial_number': 'throw_number', 'session_numeric_id': 'session_id'}, inplace=True)
                return df
            except Exception as e:
                logger.error(f"Error retrieving throws for session {session_id}: {e}")
                return pd.DataFrame()

    def get_timeseries_data(self, session_id, time_start=None, time_end=None, max_rows=None):
        numeric_id = self._resolve_session_id(session_id)
        if numeric_id is None:
            return pd.DataFrame()
        with self.db.connect() as conn:
            if not conn:
                logger.error("Failed to connect to database")
                return pd.DataFrame()
            try:
                query = "SELECT * FROM emg_timeseries WHERE session_numeric_id = %s"
                params = [numeric_id]
                if time_start is not None:
                    query += " AND time_point >= %s"
                    params.append(time_start)
                if time_end is not None:
                    query += " AND time_point <= %s"
                    params.append(time_end)
                query += " ORDER BY time_point"
                if max_rows is not None:
                    query += " LIMIT %s"
                    params.append(max_rows)
                cursor = conn.cursor(pymysql.cursors.DictCursor)
                cursor.execute(query, params)
                timeseries = cursor.fetchall()
                df = pd.DataFrame(timeseries)
                if not df.empty:
                    df.rename(columns={'session_numeric_id': 'session_id'}, inplace=True)
                logger.info(f"Fetched {len(df)} rows for session {session_id}")
                return df
            except Exception as e:
                logger.error(f"Error retrieving time series data for session {session_id}: {e}")
                return pd.DataFrame()

    def get_throw_timeseries(self, timeseries_df, throws_df):
        throw_data = {}
        if timeseries_df.empty or throws_df.empty:
            return throw_data
        for _, throw in throws_df.iterrows():
            throw_num = throw['throw_number']
            start_time, end_time = throw['start_time'], throw['end_time']
            throw_segment = timeseries_df[
                (timeseries_df['time_point'] >= start_time) & 
                (timeseries_df['time_point'] <= end_time)
            ].copy()
            throw_segment['normalized_time'] = throw_segment['time_point'] - start_time
            throw_data[throw_num] = throw_segment
        return throw_data

    def _create_plot_buffer(self, fig):
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=300)
        plt.close(fig)
        buf.seek(0)
        return buf

    def create_session_overview_plot(self, timeseries_df, throws_df, session_info):
        if timeseries_df.empty or throws_df.empty:
            logger.warning("Empty data provided for session overview plot")
            return None
        muscle1_name = session_info.get('muscle1_name', 'FCU')
        muscle2_name = session_info.get('muscle2_name', 'FCR')
        fig, ax = plt.subplots(figsize=(7, 4))  # Keeping as requested
        ax.plot(timeseries_df['time_point'], timeseries_df['muscle1_emg'], color='blue', alpha=0.7, label=muscle1_name, linewidth=0.5)
        ax.plot(timeseries_df['time_point'], timeseries_df['muscle2_emg'], color='red', alpha=0.7, label=muscle2_name, linewidth=0.5)
        for _, throw in throws_df.iterrows():
            ax.axvspan(throw['start_time'], throw['end_time'], color='green', alpha=0.2)
            ax.text(throw['start_time'], ax.get_ylim()[1] * 0.9, f"#{throw['throw_number']}", fontsize=6, ha='left')
        ax.set_title('EMG Session Overview', fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('EMG Amplitude (mV)', fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.set_xlim(timeseries_df['time_point'].min(), timeseries_df['time_point'].max())
        session_text = f"Athlete: {session_info.get('athlete_name', 'Unknown')}\nDate: {session_info.get('date_recorded', 'Unknown')}\nSession Type: {session_info.get('session_type', 'Unknown')}\nThrows: {len(throws_df)}"
        fig.text(0.02, 0.02, session_text, fontsize=6)
        return self._create_plot_buffer(fig)

    def create_metrics_evolution_plot(self, throws_df, session_info):
        if throws_df.empty:
            logger.warning("Empty throws data provided for metrics evolution plot")
            return None
        muscle1_name = session_info.get('muscle1_name', 'FCU')
        muscle2_name = session_info.get('muscle2_name', 'FCR')
        fig = plt.figure(figsize=(7, 6))
        gs = gridspec.GridSpec(3, 2, figure=fig)
        for ax, (title, ylabel, col1, col2) in zip(
            [fig.add_subplot(gs[i, j]) for i, j in [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]],
            [
                ('Peak Amplitude', 'Amplitude (mV)', 'muscle1_peak_amplitude', 'muscle2_peak_amplitude'),
                ('RMS Value', 'RMS (mV)', 'muscle1_rms_value', 'muscle2_rms_value'),
                ('Median Frequency', 'Freq (Hz)', 'muscle1_median_freq', 'muscle2_median_freq'),
                ('Rise Time', 'Time (s)', 'muscle1_rise_time', 'muscle2_rise_time'),
                ('Work Rate', 'Work Rate (mV·s/s)', 'muscle1_work_rate', 'muscle2_work_rate')
            ],
        ):
            ax.plot(throws_df['throw_number'], throws_df[col1], 'bo-', label=muscle1_name, linewidth=1)
            ax.plot(throws_df['throw_number'], throws_df[col2], 'ro-', label=muscle2_name, linewidth=1)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('Throw #', fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=6)
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(throws_df['throw_number'], throws_df['duration'], 'go-', linewidth=1)
        ax6.set_title('Throw Duration', fontsize=10)
        ax6.set_xlabel('Throw #', fontsize=8)
        ax6.set_ylabel('Duration (s)', fontsize=8)
        ax6.grid(True, alpha=0.3)
        ax6.tick_params(axis='both', which='major', labelsize=6)
        return self._create_plot_buffer(fig)

    def create_throw_thumbnails(self, throw_segments, session_info, max_throws=9):
        if not throw_segments:
            logger.warning("No throw segments provided for thumbnails")
            return None
        muscle1_name = session_info.get('muscle1_name', 'FCU')
        muscle2_name = session_info.get('muscle2_name', 'FCR')
        n_throws = min(len(throw_segments), max_throws)
        n_cols, n_rows = 3, (n_throws - 1) // 3 + 1
        fig = plt.figure(figsize=(7, n_rows * 2))
        for i, (throw_num, segment) in enumerate(sorted(throw_segments.items())[:max_throws]):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            ax.plot(segment['normalized_time'], segment['muscle1_emg'], 'b-', alpha=0.7, label=muscle1_name, linewidth=1)
            ax.plot(segment['normalized_time'], segment['muscle2_emg'], 'r-', alpha=0.7, label=muscle2_name, linewidth=1)
            ax.set_title(f"Throw #{throw_num}", fontsize=8)
            ax.set_xlabel('Time (s)', fontsize=6)
            ax.set_ylabel('EMG (mV)', fontsize=6)
            if i == 0:
                ax.legend(fontsize=6)
            stats_text = f"Peak {muscle1_name}: {segment['muscle1_emg'].max():.2f}\nPeak {muscle2_name}: {segment['muscle2_emg'].max():.2f}\nDur: {segment['normalized_time'].max():.2f}s"
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=5, va='top', bbox=dict(facecolor='white', alpha=0.7))
            ax.tick_params(axis='both', which='major', labelsize=5)
        return self._create_plot_buffer(fig)

    def get_basic_observations(self, throws_df, session_info):
        if throws_df.empty:
            return ["No throws detected in this session."]
        muscle1_name = session_info.get('muscle1_name', 'FCU')
        muscle2_name = session_info.get('muscle2_name', 'FCR')
        observations = [
            f"Total throws detected: {len(throws_df)}",
            f"Highest {muscle1_name} activity: Throw #{throws_df.loc[throws_df['muscle1_peak_amplitude'].idxmax(), 'throw_number']} ({throws_df['muscle1_peak_amplitude'].max():.2f} mV)",
            f"Highest {muscle2_name} activity: Throw #{throws_df.loc[throws_df['muscle2_peak_amplitude'].idxmax(), 'throw_number']} ({throws_df['muscle2_peak_amplitude'].max():.2f} mV)",
            f"Longest throw duration: Throw #{throws_df.loc[throws_df['duration'].idxmax(), 'throw_number']} ({throws_df['duration'].max():.2f} s)"
        ]
        for muscle, name in [('muscle1', muscle1_name), ('muscle2', muscle2_name)]:
            col = f'{muscle}_peak_amplitude'
            decreasing_count, max_decreasing, decreasing_start = 0, 0, 0
            for i in range(1, len(throws_df)):
                if throws_df.iloc[i][col] < throws_df.iloc[i-1][col]:
                    if decreasing_count == 0:
                        decreasing_start = throws_df.iloc[i-1]['throw_number']
                    decreasing_count += 1
                else:
                    max_decreasing = max(max_decreasing, decreasing_count)
                    decreasing_count = 0
            if max_decreasing >= 3:
                observations.append(f"Notable decrease in {name} activity starting at Throw #{decreasing_start}")
        for col, label in [
            ('muscle1_peak_amplitude', f"{muscle1_name} peak amplitude"),
            ('muscle2_peak_amplitude', f"{muscle2_name} peak amplitude"),
            ('muscle1_median_freq', f"{muscle1_name} median frequency"),
            ('muscle2_median_freq', f"{muscle2_name} median frequency")
        ]:
            mean = throws_df[col].mean()
            for thresh, desc in [(1.5, "high"), (0.5, "low")]:
                outliers = throws_df[throws_df[col].gt(mean * thresh) if thresh > 1 else throws_df[col].lt(mean * thresh)]
                if not outliers.empty and len(outliers) <= 3:
                    observations.append(f"Unusually {desc} {label} in Throw(s): #{', #'.join(map(str, outliers['throw_number'].tolist()))}")
        return observations

    def generate_metrics_table_data(self, throws_df, session_info):
        if throws_df.empty:
            return [["No throws detected"]]
        muscle1_name = session_info.get('muscle1_name', 'FCU')
        muscle2_name = session_info.get('muscle2_name', 'FCR')
        headers = ["Throw #", "Start (s)", "Dur (s)", f"{muscle1_name} Peak (mV)", f"{muscle2_name} Peak (mV)",
                   f"{muscle1_name} RMS (mV)", f"{muscle2_name} RMS (mV)", f"{muscle1_name} Freq (Hz)",
                   f"{muscle2_name} Freq (Hz)", f"{muscle1_name} Rise (s)", f"{muscle2_name} Rise (s)"]
        rows = [headers]
        for _, throw in throws_df.iterrows():
            rows.append([
                int(throw['throw_number']),
                round(throw['start_time'], 2),
                round(throw['duration'], 3),
                round(throw['muscle1_peak_amplitude'], 2),
                round(throw['muscle2_peak_amplitude'], 2),
                round(throw['muscle1_rms_value'], 2),
                round(throw['muscle2_rms_value'], 2),
                round(throw['muscle1_median_freq'], 1),
                round(throw['muscle2_median_freq'], 1),
                round(throw['muscle1_rise_time'], 3),
                round(throw['muscle2_rise_time'], 3)
            ])
        return rows

    def add_page_number(self, canvas, doc):
        page_num = canvas.getPageNumber()
        text = f"Page {page_num} | Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(HexColor('#4B5563'))
        canvas.drawRightString(doc.width + doc.rightMargin, 0.25*inch, text)

    def create_pdf_report(self, session_id):
        """Create a professional PDF report with distinct data branches and even layout."""
        try:
            session_info = self.get_session_info(session_id)
            if not session_info:
                logger.error(f"Session {session_id} not found")
                return None
            throws_df = self.get_throws_for_session(session_id)
            if throws_df.empty:
                logger.warning(f"No throws found for session {session_id}")
                return None
            timeseries_df = self.get_timeseries_data(session_id)
            if timeseries_df.empty:
                logger.warning(f"No time series data found for session {session_id}")
                return None
            
            throw_segments = self.get_throw_timeseries(timeseries_df, throws_df)
            overview_plot = self.create_session_overview_plot(timeseries_df, throws_df, session_info)
            metrics_plot = self.create_metrics_evolution_plot(throws_df, session_info)
            thumbnails_plot = self.create_throw_thumbnails(throw_segments, session_info)
            observations = self.get_basic_observations(throws_df, session_info)
            metrics_table_data = self.generate_metrics_table_data(throws_df, session_info)
            
            filename = session_info.get('filename', session_id)
            pdf_filename = os.path.join(self.output_dir, f"{filename}_emgreport.pdf")
            doc = SimpleDocTemplate(
                pdf_filename, 
                pagesize=letter,
                rightMargin=0.75*inch, leftMargin=0.75*inch, topMargin=1*inch, bottomMargin=0.75*inch,
                onLaterPages=self.add_page_number
            )
            elements = []

            # Cover Page (Data Branch 1)
            elements.append(Spacer(1, 1.5*inch))  # Push content down for centering
            elements.append(Paragraph("EMG Muscle Activity Report", self.styles['CoverTitle']))
            elements.append(Spacer(1, 0.5*inch))
            cover_info = f"<b>Athlete:</b> {session_info.get('athlete_name', 'Unknown')}<br/>" \
                         f"<b>TraqID:</b> {session_info.get('traq_id', 'Unknown')}<br/>" \
                         f"<b>Date:</b> {session_info.get('date_recorded', 'Unknown')}<br/>" \
                         f"<b>Session Type:</b> {session_info.get('session_type', 'Unknown')}"
            elements.append(Paragraph(cover_info, self.styles['CoverBody']))
            elements.append(PageBreak())

            # Session Overview (Data Branch 2)
            elements.append(Spacer(1, 0.5*inch))
            elements.append(Paragraph("Session Overview", self.styles['SectionTitle']))
            details = f"<b>Muscles:</b> {session_info.get('muscle1_name', 'FCU')} and {session_info.get('muscle2_name', 'FCR')} | " \
                      f"<b>Sampling Rate:</b> {session_info.get('muscle1_fs', 'Unknown')} Hz | " \
                      f"<b>Throws Detected:</b> {len(throws_df)}"
            elements.append(Paragraph(details, self.styles['SectionBody']))
            elements.append(Spacer(1, 0.25*inch))
            if overview_plot:
                img = Image(overview_plot, width=6.5*inch, height=3.5*inch)
                img.hAlign = 'CENTER'
                elements.append(img)
                elements.append(Spacer(1, 0.1*inch))
                elements.append(Paragraph("Figure 1: EMG activity across the session with throw markers", self.styles['SectionBody']))
            elements.append(PageBreak())

            # Key Observations (Data Branch 3)
            elements.append(Spacer(1, 0.5*inch))
            elements.append(Paragraph("Key Observations", self.styles['SectionTitle']))
            elements.append(Spacer(1, 0.25*inch))
            obs_elements = [Paragraph(f"• {obs}", self.styles['SectionBody']) for obs in observations]
            elements.append(KeepTogether(obs_elements))
            elements.append(PageBreak())

            # Metrics Evolution (Data Branch 4)
            elements.append(Spacer(1, 0.5*inch))
            elements.append(Paragraph("Metrics Evolution", self.styles['SectionTitle']))
            elements.append(Spacer(1, 0.25*inch))
            if metrics_plot:
                img = Image(metrics_plot, width=6.5*inch, height=4.5*inch)
                img.hAlign = 'CENTER'
                elements.append(img)
                elements.append(Spacer(1, 0.1*inch))
                elements.append(Paragraph("Figure 2: Progression of key metrics across throws", self.styles['SectionBody']))
            elements.append(PageBreak())

            # Throw Profiles (Data Branch 5)
            elements.append(Spacer(1, 0.5*inch))
            elements.append(Paragraph("Throw Profiles", self.styles['SectionTitle']))
            elements.append(Spacer(1, 0.25*inch))
            if thumbnails_plot:
                img = Image(thumbnails_plot, width=6.5*inch, height=3*inch)
                img.hAlign = 'CENTER'
                elements.append(img)
                elements.append(Spacer(1, 0.1*inch))
                elements.append(Paragraph("Figure 3: Individual throw EMG profiles (up to 9 throws)", self.styles['SectionBody']))
            elements.append(PageBreak())

            # Detailed Metrics (Data Branch 6)
            elements.append(Spacer(1, 0.5*inch))
            elements.append(Paragraph("Detailed Metrics", self.styles['SectionTitle']))
            elements.append(Spacer(1, 0.25*inch))
            col_widths = [0.5*inch, 0.7*inch, 0.5*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch]
            metrics_table = Table(metrics_table_data, colWidths=col_widths, repeatRows=1)
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#E5E7EB')),
                ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#1E3A8A')),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#D1D5DB')),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('ROWBACKGROUND', (0, 1), (-1, -1), [colors.white, HexColor('#F9FAFB')]),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('WORDWRAP', (0, 0), (-1, -1), 'WORD'),
            ]))
            elements.append(metrics_table)

            doc.build(elements)
            logger.info(f"PDF report generated: {pdf_filename}")
            return pdf_filename
        
        except Exception as e:
            logger.error(f"Error creating PDF report for session {session_id}: {e}", exc_info=True)
            return None

    def generate_report_for_session(self, session_id):
        logger.info(f"Generating report for session {session_id}")
        return self.create_pdf_report(session_id)

    def generate_reports_for_athlete(self, athlete_name=None, traq_id=None, limit=5):
        with self.db.connect() as conn:
            if not conn:
                logger.error("Failed to connect to database")
                return []
            try:
                query = "SELECT session_id FROM emg_sessions WHERE 1=1"
                params = []
                if athlete_name:
                    query += " AND athlete_name LIKE %s"
                    params.append(f"%{athlete_name}%")
                if traq_id:
                    query += " AND traq_id = %s"
                    params.append(traq_id)
                query += " ORDER BY date_recorded DESC LIMIT %s"
                params.append(limit)
                cursor = conn.cursor()
                cursor.execute(query, params)
                sessions = cursor.fetchall()
                return [path for session in sessions if (path := self.generate_report_for_session(session[0]))]
            except Exception as e:
                logger.error(f"Error generating reports for athlete: {e}")
                return []

    def generate_reports_for_date_range(self, start_date, end_date, limit=10):
        with self.db.connect() as conn:
            if not conn:
                logger.error("Failed to connect to database")
                return []
            try:
                start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date() if isinstance(start_date, str) else start_date
                end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date() if isinstance(end_date, str) else end_date
                query = "SELECT session_id FROM emg_sessions WHERE date_recorded BETWEEN %s AND %s ORDER BY date_recorded LIMIT %s"
                cursor = conn.cursor()
                cursor.execute(query, (start_date, end_date, limit))
                sessions = cursor.fetchall()
                return [path for session in sessions if (path := self.generate_report_for_session(session[0]))]
            except Exception as e:
                logger.error(f"Error generating reports for date range: {e}")
                return []

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if len(sys.argv) < 2:
        print("Usage: python emg_report_generator.py [session_id]")
        sys.exit(1)
    session_id = sys.argv[1]
    output_dir = r"C:\Users\alex.britton\Documents\DelsysTesting\Reports"
    load_dotenv()
    db_config = {
        'host': os.getenv('DB_HOST'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'database': os.getenv('DB_NAME')
    }
    generator = EMGReportGenerator(db_config=db_config, output_dir=output_dir)
    report_path = generator.generate_report_for_session(session_id)
    print(f"Report generated successfully: {report_path}" if report_path else "Failed to generate report")