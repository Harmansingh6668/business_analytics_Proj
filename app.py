import os
import io
import base64
import shutil
import pandas as pd
import numpy as np
# Set matplotlib backend to Agg before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_file, jsonify
from sklearn.cluster import KMeans
from reportlab.lib.pagesizes import portrait, A4
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
import plotly.express as px
import json


app = Flask(__name__)

ALLOWED_EXTENSIONS = {'csv', 'txt', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('start.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/team')
def team():
    return render_template('team.html')


@app.route('/under_construct')
def under_constr():
    return render_template('under_constr.html')


@app.route('/wifi-tunneling')
def wifi_tunneling():
    return render_template('wifi_tunneling.html')


@app.route('/readme')
def readme():
    return render_template('readme.html')


@app.route('/index')
def index():
    folder_path = 'static/plots'
    os.makedirs(folder_path, exist_ok=True)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            os.remove(file_path)
        except Exception:
            pass
    return render_template('form-cs.html')


@app.route('/generate-clusters', methods=['POST'])
def generate_clusters():
    try:
        num_clusters = int(request.form['num_clusters'])
        algo_type = request.form['algo_type']
        dataset_selection = request.form['dataset']

        os.makedirs('datasets', exist_ok=True)

        # Handle dataset selection
        if dataset_selection == 'custom':
            file = request.files.get('custom_dataset')
            if not (file and allowed_file(file.filename)):
                return render_template('error.html', message="Invalid or missing custom file.")
            dataset_path = os.path.join('datasets', 'custom.csv')
            file.save(dataset_path)
        else:
            dataset_map = {
                'dataset1': 'datasets/Sales Transaction.csv'
            }
            dataset_path = dataset_map.get(dataset_selection)
            if not dataset_path or not os.path.exists(dataset_path):
                return render_template('error.html', message="Dataset not found.")

        # Load dataset
        dataset = pd.read_csv(dataset_path)
        dataset = dataset.select_dtypes(include=[np.number]).dropna()

        if dataset.empty:
            return render_template('error.html', message="Dataset has no numeric data.")

        dataset = dataset.astype(float)
        feature_names = dataset.columns.tolist()
        table_html = dataset.head(10).to_html(classes="table table-striped", index=False)

        # Run K-Means
        if algo_type == 'k-means':
            kmeans = KMeans(n_clusters=num_clusters, n_init=10)
            dataset['cluster'] = kmeans.fit_predict(dataset)

            plot_dir = 'static/plots'
            os.makedirs(plot_dir, exist_ok=True)
            image_names = []

            features = [col for col in dataset.columns if col != 'cluster']
            for i in range(len(features)):
                for j in range(i + 1, len(features)):
                    plt.figure(figsize=(6, 4))
                    plt.scatter(dataset[features[i]], dataset[features[j]], c=dataset['cluster'], cmap='tab10')
                    plt.xlabel(features[i])
                    plt.ylabel(features[j])
                    plt.tight_layout()
                    filename = f"plot_{features[i]}_{features[j]}.png"
                    plt.savefig(os.path.join(plot_dir, filename))
                    plt.close()
                    image_names.append(filename)

            return render_template('result.html',
                                   image_names=image_names,
                                   feature_names=features,
                                   table_html=table_html)

        else:
            return render_template('under_constr.html')

    except Exception as e:
        return render_template('error.html', message=str(e))


@app.route('/download-pdf')
def download_pdf():
    try:
        image_folder = 'static/plots'
        pdf_path = 'static/pdf/report.pdf'
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

        images = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg'))]
        if not images:
            return render_template('error.html', message="No images to export.")

        pdf_canvas = canvas.Canvas(pdf_path, pagesize=portrait(A4))

        # Cover page
        intro_img = 'static/intro.png'
        if os.path.exists(intro_img):
            pdf_canvas.drawImage(intro_img, 0, 0, width=A4[0], height=A4[1])
            pdf_canvas.showPage()

        # Image pages
        y_start = A4[1] - 1.5 * inch
        for i, image in enumerate(images):
            img_path = os.path.join(image_folder, image)
            pdf_canvas.drawImage(img_path, 1 * inch, y_start - 5 * inch, width=5.5 * inch, height=4.5 * inch)
            pdf_canvas.drawCentredString(A4[0] / 2, 0.5 * inch, image)
            pdf_canvas.showPage()

        # Ending page
        end_img = 'static/end.png'
        if os.path.exists(end_img):
            pdf_canvas.drawImage(end_img, 0, 0, width=A4[0], height=A4[1])

        pdf_canvas.save()
        return send_file(pdf_path, as_attachment=True)

    except Exception as e:
        return render_template('error.html', message=str(e))


def convert_plot_to_html(fig):
    """Convert a matplotlib figure to HTML string"""
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return f'<img src="data:image/png;base64,{base64.b64encode(img.getvalue()).decode()}" class="img-fluid">'


@app.route('/tableau')
def tableau():
    try:
        # Get selected dataset from query parameter, default to 'sales'
        selected_dataset = request.args.get('dataset', 'sales')
        
        # Map dataset selection to file paths
        dataset_map = {
            'sales': 'Sales Transaction.csv',
            'geolocation': 'geolocation.csv',
            'sports': 'Sports Performance.csv',
            'wine': 'Wine Quality.csv'
        }
        
        # Get the dataset path and verify it exists
        dataset_path = os.path.join(os.path.dirname(__file__), 'datasets', dataset_map.get(selected_dataset, dataset_map['sales']))
        
        if not os.path.exists(dataset_path):
            return render_template('error.html', 
                                message=f"Dataset file not found. Please ensure {dataset_map.get(selected_dataset)} exists in the datasets folder.")
        
        # Read the dataset
        df = pd.read_csv(dataset_path)
        
        if selected_dataset == 'sports':
            # Special handling for sports dataset
            # Age distribution plot
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            fig1.patch.set_facecolor('#f8f9fa')
            ax1.set_facecolor('#ffffff')
            
            # Create age bins
            age_bins = list(range(10, 80, 10))
            ax1.hist(df['Age'], bins=age_bins, alpha=0.6, color='#4b6cb7', edgecolor='white')
            ax1.set_xlabel('Age Groups', fontsize=10)
            ax1.set_ylabel('Number of Participants', fontsize=10)
            ax1.set_title('Age Distribution in Sports', fontsize=12, pad=20)
            ax1.grid(True, alpha=0.3)
            scatter_plot = convert_plot_to_html(fig1)
            plt.close(fig1)
            
            # Sports participation distribution
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            fig2.patch.set_facecolor('#f8f9fa')
            ax2.set_facecolor('#ffffff')
            
            sports_count = df['Sports'].value_counts()
            bars = ax2.bar(sports_count.index, sports_count.values, color='#4b6cb7', alpha=0.6)
            ax2.set_xlabel('Sports', fontsize=10)
            ax2.set_ylabel('Number of Participants', fontsize=10)
            ax2.set_title('Sports Participation Distribution', fontsize=12, pad=20)
            plt.xticks(rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on top of each bar
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom')
            
            dist_plot = convert_plot_to_html(fig2)
            plt.close(fig2)
            
            # Generate statistics table
            stats_table = f"""
            <table class='table table-striped table-hover table-bordered'>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Participants</td><td>{len(df)}</td></tr>
                <tr><td>Average Age</td><td>{df['Age'].mean():.2f}</td></tr>
                <tr><td>Most Popular Sport</td><td>{df['Sports'].mode().iloc[0]}</td></tr>
                <tr><td>Age Range</td><td>{df['Age'].min():.0f} - {df['Age'].max():.0f}</td></tr>
                <tr><td>Gender Distribution</td><td>Male: {len(df[df['Gender']=='male'])}, Female: {len(df[df['Gender']=='female'])}</td></tr>
            </table>
            """
            
            return render_template('tableau.html',
                             selected_dataset=selected_dataset,
                             scatter_plot=scatter_plot,
                             dist_plot=dist_plot,
                             stats_table=stats_table)
                             
        else:
            # Default handling for other datasets
            numeric_cols = df.select_dtypes(include=[np.number])
            if numeric_cols.empty:
                return render_template('error.html', 
                                    message="No numeric columns found in the dataset for visualization.")
            
            # Create plots
            plt.style.use('classic')
            
            # Scatter plot of first two numeric columns
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            fig1.patch.set_facecolor('#f8f9fa')
            ax1.set_facecolor('#ffffff')
            
            x_col = numeric_cols.columns[0]
            y_col = numeric_cols.columns[1] if len(numeric_cols.columns) > 1 else numeric_cols.columns[0]
            ax1.scatter(numeric_cols[x_col], numeric_cols[y_col], alpha=0.6, color='#4b6cb7')
        ax1.set_xlabel(x_col)
        ax1.set_ylabel(y_col)
        ax1.set_title(f'{x_col} vs {y_col}')
        scatter_plot = convert_plot_to_html(fig1)
        plt.close(fig1)
        
        # Distribution plot of first numeric column
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        fig2.patch.set_facecolor('#f8f9fa')  # Light gray background
        ax2.set_facecolor('#ffffff')  # White plot background
        
        ax2.hist(numeric_cols[x_col], bins=30, color='#4b6cb7', alpha=0.6, edgecolor='white')
        ax2.set_xlabel(x_col, fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.set_title(f'Distribution of {x_col}', fontsize=12, pad=20)
        ax2.grid(True, alpha=0.3)
        dist_plot = convert_plot_to_html(fig2)
        plt.close(fig2)
        
        # Summary statistics table
        stats_table = df.describe().to_html(
            classes="table table-striped table-hover table-bordered",
            float_format=lambda x: '{:.2f}'.format(x) if isinstance(x, (float, np.floating)) else x
        )
        
        return render_template('tableau.html',
                             selected_dataset=selected_dataset,
                             scatter_plot=scatter_plot,
                             dist_plot=dist_plot,
                             stats_table=stats_table)
                             
    except Exception as e:
        import traceback
        error_msg = f"An error occurred: {str(e)}\n{traceback.format_exc()}"
        return render_template('error.html', message=error_msg)


if __name__ == '__main__':
    app.run(debug=True)

