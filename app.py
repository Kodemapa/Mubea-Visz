from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
import h5py
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple
import glob

# Try to import dotenv, but don't fail if it's not available
try:
    from dotenv import load_dotenv
    load_dotenv()
    dotenv_available = True
except ImportError:
    dotenv_available = False
    print("python-dotenv not installed. Using default configuration values.")

class Config:
    """Configuration class to store application settings"""
    def __init__(self):
        # Data dimensions
        self.data_rows = self._get_env_int('DATA_ROWS', 100)
        self.data_points = self._get_env_int('DATA_POINTS', 41)
        self.ref_points = self._get_env_int('REF_POINTS', 21)
        
        # Reference data parameters
        self.x_max = self._get_env_float('X_MAX', 1600)
        self.ref_cycles = self._get_env_float('REF_CYCLES', 8)
        self.amplitude = self._get_env_float('AMPLITUDE', 1000)
        self.offset = self._get_env_float('OFFSET', 1000)
        self.profile_scale = self._get_env_float('PROFILE_SCALE', 2000)
        
        # Noise parameters
        self.x_noise_scale = self._get_env_float('X_NOISE_SCALE', 20)
        self.z_noise_scale = self._get_env_float('Z_NOISE_SCALE', 100)
        
        # App settings
        self.secret_key = os.getenv('SECRET_KEY', 'h5_visualization_dashboard')
        self.h5_files_dir = os.getenv('H5_FILES_DIR', './data')
    
    def _get_env_int(self, name, default):
        """Get an integer from environment variables with a default value"""
        value = os.getenv(name)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            print(f"Warning: {name} environment variable is not a valid integer. Using default: {default}")
            return default
    
    def _get_env_float(self, name, default):
        """Get a float from environment variables with a default value"""
        value = os.getenv(name)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            print(f"Warning: {name} environment variable is not a valid float. Using default: {default}")
            return default

# Create config instance
config = Config()

# Flask App
app = Flask(__name__)
app.secret_key = config.secret_key

# Directory where H5 files are stored
if not os.path.exists(config.h5_files_dir):
    os.makedirs(config.h5_files_dir)

# Global data storage
data_store = {}

# Dash App for plotting
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/plot/')
dash_app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='dash-content')
])

# Helper function to load data from H5 file
def load_data_from_h5(file_path):
    """Load all data from H5 file and prepare it for display"""
    global data_store

    try:
        with h5py.File(file_path, "r") as h5_file:
            # Load screwdown data
            screw_actual_x = h5_file["process data/screwdown/x"][:]
            screw_actual_z = h5_file["process data/screwdown/z"][:]
            screw_ref_x = h5_file["process data/screwdown"].attrs["Screwdown ref x"]
            screw_ref_z = h5_file["process data/screwdown"].attrs["Screwdown ref z"]

            # Load bending data
            bend_actual_x = h5_file["process data/bending/x"][:]
            bend_actual_z = h5_file["process data/bending/z"][:]
            bend_ref_x = h5_file["process data/bending"].attrs["Bending ref x"]
            bend_ref_z = h5_file["process data/bending"].attrs["Bending ref z"]

            # Load profile data
            profile_actual_x = h5_file["process data/profile/x"][:]
            profile_actual_z = h5_file["process data/profile/z"][:]
            profile_ref_x = h5_file["process data/profile"].attrs["Profile ref x"]
            profile_ref_z = h5_file["process data/profile"].attrs["Profile ref z"]

            # Load blank info data
            blank_infos_single = h5_file["process data/blank infos/single"][:]
            blank_infos_0 = blank_infos_single[0, :config.data_rows]

        # Resize arrays to ensure consistent dimensions
        screw_actual_x = np.resize(screw_actual_x, (config.data_rows, config.data_points))
        screw_actual_z = np.resize(screw_actual_z, (config.data_rows, config.data_points))
        bend_actual_x = np.resize(bend_actual_x, (config.data_rows, config.data_points))
        bend_actual_z = np.resize(bend_actual_z, (config.data_rows, config.data_points))
        profile_actual_x = np.resize(profile_actual_x, (config.data_rows, config.data_points))
        profile_actual_z = np.resize(profile_actual_z, (config.data_rows, config.data_points))

        # Generate reference display points with midpoints
        screw_disp_x, screw_disp_z, screw_mid = generate_reference_display(screw_ref_x, screw_ref_z)
        bend_disp_x, bend_disp_z, bend_mid = generate_reference_display(bend_ref_x, bend_ref_z)
        profile_disp_x, profile_disp_z, profile_mid = generate_reference_display(profile_ref_x, profile_ref_z)

        # Store data in global data store
        data_store = {
            "screwdown": {
                "label": "Screwdown", 
                "actual_x": screw_actual_x, 
                "actual_z": screw_actual_z, 
                "ref_x": screw_disp_x, 
                "ref_z": screw_disp_z,
                "is_midpoint": screw_mid
            },
            "bending": {
                "label": "Bending", 
                "actual_x": bend_actual_x, 
                "actual_z": bend_actual_z, 
                "ref_x": bend_disp_x, 
                "ref_z": bend_disp_z,
                "is_midpoint": bend_mid
            },
            "profile": {
                "label": "Profile", 
                "actual_x": profile_actual_x, 
                "actual_z": profile_actual_z, 
                "ref_x": profile_disp_x, 
                "ref_z": profile_disp_z,
                "is_midpoint": profile_mid
            },
            "blank_info": {
                "label": "Blank Info",
                "data": blank_infos_0
            }
        }
        
        return True
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return False

def generate_reference_display(ref_x, ref_z):
    """Generate reference display points with midpoints"""
    display_x, display_z, is_midpoint = [], [], []

    for i in range(len(ref_x) - 1):
        display_x.append(ref_x[i])
        display_z.append(ref_z[i])
        is_midpoint.append(False)
        
        # Calculate midpoint
        mx = (ref_x[i] + ref_x[i+1]) / 2
        mz = (ref_z[i] + ref_z[i+1]) / 2
        display_x.append(mx)
        display_z.append(mz)
        is_midpoint.append(True)

    # Add last point
    display_x.append(ref_x[-1])
    display_z.append(ref_z[-1])
    is_midpoint.append(False)

    # Limit to data_points points
    return display_x[:config.data_points], display_z[:config.data_points], is_midpoint[:config.data_points]

# Routes
@app.route('/')
def index():
    """Home page - List all H5 files"""
    # Get all H5 files in the directory
    h5_files = glob.glob(os.path.join(config.h5_files_dir, '*.h5'))
    h5_files = [os.path.basename(file) for file in h5_files]

    # If no files found, add a sample file for demonstration
    if not h5_files:
        h5_files = ['test.h5', 'sample1.h5', 'sample2.h5', 'sample3.h5', 'sample4.h5']

    return render_template('index.html', h5_files=h5_files)

@app.route('/select_database', methods=['GET', 'POST'])
def select_database():
    """Page to select which database to visualize"""
    if request.method == 'POST':
        selected_file = request.form.get('selected_file')
        if selected_file:
            session['selected_file'] = selected_file
            
            # Load data from the selected file
            file_path = os.path.join(config.h5_files_dir, selected_file)
            if os.path.exists(file_path):
                load_data_from_h5(file_path)
            else:
                # For demo purposes, load from test.h5 if available, otherwise create sample data
                if os.path.exists('test.h5'):
                    load_data_from_h5('test.h5')
                else:
                    # Create sample data
                    create_sample_data()
            
            return render_template('select_database.html')

    # If no file was selected, redirect to home
    if 'selected_file' not in session:
        return redirect(url_for('index'))

    return render_template('select_database.html')

@app.route('/visualize', methods=['GET', 'POST'])
def visualize():
    """Page to visualize the selected database"""
    if request.method == 'POST':
        selected_db = request.form.get('selected_db')
        if selected_db:
            session['selected_db'] = selected_db

    # If no database was selected, redirect to database selection
    if 'selected_db' not in session or 'selected_file' not in session:
        return redirect(url_for('select_database'))

    selected_file = session['selected_file']
    selected_db = session['selected_db']

    # Ensure data is loaded
    if not data_store:
        file_path = os.path.join(config.h5_files_dir, selected_file)
        if os.path.exists(file_path):
            load_data_from_h5(file_path)
        else:
            # For demo purposes, load from test.h5 if available, otherwise create sample data
            if os.path.exists('test.h5'):
                load_data_from_h5('test.h5')
            else:
                # Create sample data
                create_sample_data()

    return render_template('visualize.html', 
                          selected_file=selected_file,
                          selected_db=selected_db)

@app.route('/get_data')
def get_data():
    """API endpoint to get data for the table"""
    if 'selected_db' not in session:
        return jsonify({'error': 'No database selected'})

    selected_db = session['selected_db']

    # Ensure data is loaded
    if not data_store:
        if 'selected_file' in session:
            file_path = os.path.join(config.h5_files_dir, session['selected_file'])
            if os.path.exists(file_path):
                load_data_from_h5(file_path)
            else:
                # For demo purposes, load from test.h5 if available, otherwise create sample data
                if os.path.exists('test.h5'):
                    load_data_from_h5('test.h5')
                else:
                    # Create sample data
                    create_sample_data()

    try:
        # Prepare data for JSON response - convert NumPy types to Python native types
        if selected_db == 'blank_info':
            return jsonify({
                'blank_info': data_store['blank_info']['data'].tolist()
            })
        else:
            # Convert NumPy arrays to Python lists and NumPy values to Python native types
            ref_x = [float(x) for x in data_store[selected_db]['ref_x']]
            ref_z = [float(z) for z in data_store[selected_db]['ref_z']]
            is_midpoint = [bool(m) for m in data_store[selected_db]['is_midpoint']]
            
            return jsonify({
                'label': data_store[selected_db]['label'],
                'ref_x': ref_x,
                'ref_z': ref_z,
                'is_midpoint': is_midpoint,
                'actual_x': data_store[selected_db]['actual_x'].tolist(),
                'actual_z': data_store[selected_db]['actual_z'].tolist(),
                'blank_info': data_store['blank_info']['data'].tolist()
            })
    except Exception as e:
        return jsonify({'error': str(e)})

def create_sample_data():
    """Create sample data for demonstration"""
    global data_store

    # Create sample reference data
    screw_ref_x = np.linspace(0, config.x_max, config.ref_points)
    screw_ref_z = np.sin(np.linspace(0, config.ref_cycles*np.pi, config.ref_points)) * config.amplitude + config.offset

    bend_ref_x = np.linspace(0, config.x_max, config.ref_points)
    bend_ref_z = np.cos(np.linspace(0, config.ref_cycles*np.pi, config.ref_points)) * config.amplitude + config.offset

    profile_ref_x = np.linspace(0, config.x_max, config.ref_points)
    profile_ref_z = np.abs(np.sin(np.linspace(0, config.ref_cycles/2*np.pi, config.ref_points))) * config.profile_scale

    # Create sample actual data (with some noise)
    blank_infos_0 = np.arange(1, config.data_rows + 1)

    screw_actual_x = np.zeros((config.data_rows, config.data_points))
    screw_actual_z = np.zeros((config.data_rows, config.data_points))
    bend_actual_x = np.zeros((config.data_rows, config.data_points))
    bend_actual_z = np.zeros((config.data_rows, config.data_points))
    profile_actual_x = np.zeros((config.data_rows, config.data_points))
    profile_actual_z = np.zeros((config.data_rows, config.data_points))

    # Generate reference display points with midpoints
    screw_disp_x, screw_disp_z, screw_mid = generate_reference_display(screw_ref_x, screw_ref_z)
    bend_disp_x, bend_disp_z, bend_mid = generate_reference_display(bend_ref_x, bend_ref_z)
    profile_disp_x, profile_disp_z, profile_mid = generate_reference_display(profile_ref_x, profile_ref_z)

    # Fill actual data with reference + noise
    for i in range(config.data_rows):
        screw_noise_x = np.random.normal(0, config.x_noise_scale, config.data_points)
        screw_noise_z = np.random.normal(0, config.z_noise_scale, config.data_points)
        bend_noise_x = np.random.normal(0, config.x_noise_scale, config.data_points)
        bend_noise_z = np.random.normal(0, config.z_noise_scale, config.data_points)
        profile_noise_x = np.random.normal(0, config.x_noise_scale, config.data_points)
        profile_noise_z = np.random.normal(0, config.z_noise_scale, config.data_points)
        
        for j in range(config.data_points):
            idx = min(j, len(screw_disp_x)-1)
            screw_actual_x[i, j] = screw_disp_x[idx] + screw_noise_x[j]
            screw_actual_z[i, j] = screw_disp_z[idx] + screw_noise_z[j]
            
            idx = min(j, len(bend_disp_x)-1)
            bend_actual_x[i, j] = bend_disp_x[idx] + bend_noise_x[j]
            bend_actual_z[i, j] = bend_disp_z[idx] + bend_noise_z[j]
            
            idx = min(j, len(profile_disp_x)-1)
            profile_actual_x[i, j] = profile_disp_x[idx] + profile_noise_x[j]
            profile_actual_z[i, j] = profile_disp_z[idx] + profile_noise_z[j]

    # Store data in global data store
    data_store = {
        "screwdown": {
            "label": "Screwdown", 
            "actual_x": screw_actual_x, 
            "actual_z": screw_actual_z, 
            "ref_x": screw_disp_x, 
            "ref_z": screw_disp_z,
            "is_midpoint": screw_mid
        },
        "bending": {
            "label": "Bending", 
            "actual_x": bend_actual_x, 
            "actual_z": bend_actual_z, 
            "ref_x": bend_disp_x, 
            "ref_z": bend_disp_z,
            "is_midpoint": bend_mid
        },
        "profile": {
            "label": "Profile", 
            "actual_x": profile_actual_x, 
            "actual_z": profile_actual_z, 
            "ref_x": profile_disp_x, 
            "ref_z": profile_disp_z,
            "is_midpoint": profile_mid
        },
        "blank_info": {
            "label": "Blank Info",
            "data": blank_infos_0
        }
    }

# Dash callback for plotting
@dash_app.callback(
    Output('dash-content', 'children'),
    Input('url', 'pathname')
)
def update_dash_content(pathname):
    if not pathname or pathname == '/plot/':
        return html.Div("Please select a row to display the graph")

    try:
        # Extract row_id from pathname
        parts = pathname.strip('/').split('/')
        if len(parts) < 2:
            return html.Div("Invalid URL format")
        
        row_id = int(parts[1])
        db_type = session.get('selected_db', 'screwdown')
        
        # Ensure data is loaded
        if not data_store:
            if 'selected_file' in session:
                file_path = os.path.join(config.h5_files_dir, session['selected_file'])
                if os.path.exists(file_path):
                    load_data_from_h5(file_path)
                else:
                    # For demo purposes, load from test.h5 if available, otherwise create sample data
                    if os.path.exists('test.h5'):
                        load_data_from_h5('test.h5')
                    else:
                        # Create sample data
                        create_sample_data()
        
        # Helper function to format numbers (remove trailing zeros)
        def format_number(num):
            if num == int(num):
                return str(int(num))
            return str(num).rstrip('0').rstrip('.') if '.' in str(num) else str(num)
        
        # Get data for the selected row and database
        if db_type in ['screwdown', 'bending', 'profile']:
            # Convert NumPy arrays to Python lists for Plotly
            actual_x = data_store[db_type]['actual_x'][row_id].tolist()
            actual_z = data_store[db_type]['actual_z'][row_id].tolist()
            ref_x = [float(x) for x in data_store[db_type]['ref_x']]
            ref_z = [float(z) for z in data_store[db_type]['ref_z']]
            label = data_store[db_type]['label']
            
            # Create figure
            fig = go.Figure()
            
            # Add reference line
            fig.add_trace(go.Scatter(
                x=ref_x, 
                y=ref_z, 
                mode='lines', 
                name=f'{label} Reference',
                text=[f"({format_number(x)}, {format_number(z)})" for x, z in zip(ref_x, ref_z)], 
                hoverinfo='text',
                line=dict(color='#0d6efd', width=2)
            ))
            
            # Add actual line with markers
            fig.add_trace(go.Scatter(
                x=actual_x, 
                y=actual_z, 
                mode='lines+markers', 
                name=f'{label} Actual',
                text=[f"({format_number(x)}, {format_number(z)})" for x, z in zip(actual_x, actual_z)], 
                hoverinfo='text',
                line=dict(color='#198754', width=2),
                marker=dict(size=8)
            ))
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text=f'{label} Actual vs Reference (Row {row_id})',
                    font=dict(size=18)
                ),
                xaxis=dict(
                    title='X Position',
                    gridcolor='#eee',
                    zeroline=False
                ),
                yaxis=dict(
                    title='Z Position',
                    gridcolor='#eee',
                    zeroline=False
                ),
                hovermode='closest',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
                margin=dict(l=50, r=50, t=80, b=50),
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Arial, sans-serif"),
                autosize=True
            )
            
            return dcc.Graph(figure=fig, style={'height': '100vh'})
        
        elif db_type == 'blank_info':
            # For blank_info, just show a bar chart of the values
            blank_info = data_store['blank_info']['data'].tolist()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(range(len(blank_info))),
                y=blank_info,
                text=[format_number(val) for val in blank_info],
                hoverinfo='text',
                marker_color='#0d6efd'
            ))
            
            fig.update_layout(
                title=dict(
                    text=f'Blank Info Values',
                    font=dict(size=18)
                ),
                xaxis=dict(
                    title='Index',
                    gridcolor='#eee',
                    zeroline=False
                ),
                yaxis=dict(
                    title='Value',
                    gridcolor='#eee',
                    zeroline=False
                ),
                margin=dict(l=50, r=50, t=80, b=50),
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Arial, sans-serif"),
                autosize=True
            )
            
            return dcc.Graph(figure=fig, style={'height': '100vh'})
        
    except Exception as e:
        return html.Div(f"Error generating graph: {str(e)}")

    return html.Div("No data to display")

if __name__ == '__main__':
    # Try to load data from test.h5 if it exists
    if os.path.exists('test.h5'):
        load_data_from_h5('test.h5')

    # Run the application
    run_simple('localhost', 5000, app, use_reloader=True, use_debugger=True)

