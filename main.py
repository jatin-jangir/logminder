import kubernetes
import time
import datetime
from kubernetes import client, config
import psycopg2
import os
from datetime import timezone

# Initialize Kubernetes API client
try:
    config.load_incluster_config()
    print("Loaded in-cluster Kubernetes config")
except config.ConfigException:
    config.load_kube_config()
    print("Loaded local kubeconfig")

v1 = client.CoreV1Api()

# PostgreSQL connection setup
def get_postgres_connection():
    try:
        conn = psycopg2.connect(
            host="127.0.0.1",
            database="logminder",  # Replace with your PostgreSQL database name
            user="postgres",
            password=os.getenv('POSTGRES_PASSWORD'),  # Use environment variable for password
            port="5432"
        )
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None

# Function to get logs from a container
def get_logs(namespace, pod_name, container_name, since_seconds):
    try:
        logs = v1.read_namespaced_pod_log(
            name=pod_name,
            namespace=namespace,
            container=container_name,
            since_seconds=since_seconds
        )
        return logs.splitlines()
    except client.exceptions.ApiException as e:
        print(f"Error fetching logs: {e}")
        return []

# Function to count lines with and without "error"
def count_log_lines(log_lines):
    error_count = 0
    non_error_count = 0
    
    for line in log_lines:
        if "warn" in line.lower():
            error_count += 1
        else:
            non_error_count += 1
    
    return error_count, non_error_count

# Function to store time-series data in PostgreSQL
def store_time_series_data(namespace, pod_name, container_name, timestamp, error_count, non_error_count):
    time_series_data = {
        "timestamp": timestamp,
        "namespace": namespace,
        "pod": pod_name,
        "container": container_name,
        "error_count": error_count,
        "non_error_count": non_error_count
    }
    
    conn = get_postgres_connection()
    if conn is None:
        return
    
    try:
        cur = conn.cursor()
        insert_query = """
        INSERT INTO log_metrics (timestamp, namespace, pod, container, error_count, non_error_count)
        VALUES (%s, %s, %s, %s, %s, %s);
        """
        cur.execute(insert_query, (
            timestamp,
            namespace,
            pod_name,
            container_name,
            error_count,
            non_error_count
        ))
        conn.commit()
        cur.close()
        print(f"Stored in PostgreSQL: {time_series_data}")
    except psycopg2.Error as e:
        print(f"Error inserting data into PostgreSQL: {e}")
    finally:
        conn.close()

# Main function to monitor logs and count "error" occurrences every minute
def monitor_pods():
    namespace = "test-ns"  # Set your desired namespace
    while True:
        pods = v1.list_namespaced_pod(namespace)
        
        for pod in pods.items:
            pod_name = pod.metadata.name
            for container in pod.spec.containers:
                container_name = container.name
                current_time = datetime.datetime.now(timezone.utc)
                since_time = current_time - datetime.timedelta(minutes=1)
                
                # Calculate the time difference in seconds
                since_seconds = int((current_time - since_time).total_seconds())
                
                # Get logs from the last minute
                logs = get_logs(namespace, pod_name, container_name, since_seconds)
                error_count, non_error_count = count_log_lines(logs)
                
                # Store the count in PostgreSQL
                store_time_series_data(
                    namespace, pod_name, container_name, current_time.isoformat(),
                    error_count, non_error_count
                )
        
        time.sleep(60)  # Sleep for 1 minute before next iteration

if __name__ == "__main__":
    monitor_pods()
