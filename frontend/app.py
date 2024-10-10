import os
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta

app = Flask(__name__)

# Database configuration using environment variable for the password
app.config['SQLALCHEMY_DATABASE_URI'] = f"postgresql://postgres:{os.getenv('POSTGRES_PASSWORD')}@127.0.0.1:5432/logminder"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class LogMetrics(db.Model):
    __tablename__ = 'log_metrics'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime)
    namespace = db.Column(db.String(50))
    pod = db.Column(db.String(50))
    container = db.Column(db.String(50))
    error_count = db.Column(db.Integer)
    non_error_count = db.Column(db.Integer)

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Fetch namespaces
@app.route('/get_namespaces', methods=['GET'])
def get_namespaces():
    namespaces = db.session.query(LogMetrics.namespace).distinct().all()
    namespaces = [n[0] for n in namespaces]
    return jsonify(namespaces)

# Fetch containers for a namespace
@app.route('/get_containers', methods=['GET'])
def get_containers():
    namespace = request.args.get('namespace')
    containers = db.session.query(LogMetrics.container).filter_by(namespace=namespace).distinct().all()
    containers = [c[0] for c in containers]
    return jsonify(containers)

# Fetch log metrics based on selected options
@app.route('/get_metrics', methods=['GET'])
def get_metrics():
    namespace = request.args.get('namespace')
    container = request.args.get('container')
    time_range = int(request.args.get('time_range', 10))  # in minutes, default 10 min

    time_limit = datetime.utcnow() - timedelta(minutes=time_range)
    metrics = db.session.query(LogMetrics).filter(
        LogMetrics.namespace == namespace,
        LogMetrics.container == container,
        LogMetrics.timestamp >= time_limit
    ).order_by(LogMetrics.timestamp).all()

    data = {
        'timestamps': [m.timestamp.strftime('%Y-%m-%d %H:%M:%S') for m in metrics],
        'error_counts': [m.error_count for m in metrics],
        'non_error_counts': [m.non_error_count for m in metrics]
    }
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)
