from flask import Flask, render_template, send_from_directory, abort
import os

app = Flask(__name__)

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/analyze')
def analyze():
    platforms = ['amazon', 'bestbuy', 'ebay', 'flipkart', 'walmart', 'sixth']  # Updated to include sixth
    return render_template('analysis.html', platforms=platforms)

@app.route('/analyze/<platform>')
def show_analysis(platform):
    if platform not in ['amazon', 'bestbuy', 'ebay', 'flipkart', 'walmart', 'sixth']:
        abort(404)
    return render_template('analysis.html', platform=platform)

@app.route('/compare')
def compare():
    return render_template('comparison.html')

@app.route('/download_report')
def download_report():
    try:
        return send_from_directory('static/graphs', 'sentiment_report.csv', as_attachment=True)
    except FileNotFoundError:
        abort(404, description="Report file not found. Please run preprocess.py first.")

@app.route('/download_platform_report/<platform>')
def download_platform_report(platform):
    if platform not in ['amazon', 'bestbuy', 'ebay', 'flipkart', 'walmart', 'sixth']:
        abort(404)
    try:
        return send_from_directory('static/graphs', f'{platform}_report.csv', as_attachment=True)
    except FileNotFoundError:
        abort(404, description=f"{platform}_report.csv not found. Please run preprocess.py first.")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)