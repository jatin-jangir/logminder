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

# Function to get the last stored timestamp for a specific namespace, pod, and container
def get_last_stored_time(namespace, pod_name, container_name):
    conn = get_postgres_connection()
    if conn is None:
        return None
    
    try:
        cur = conn.cursor()
        select_query = """
        SELECT MAX(timestamp) FROM log_metrics 
        WHERE namespace = %s AND pod = %s AND container = %s;
        """
        cur.execute(select_query, (namespace, pod_name, container_name))
        last_stored_time = cur.fetchone()[0]
        cur.close()
        return last_stored_time
    except psycopg2.Error as e:
        print(f"Error fetching last stored time from PostgreSQL: {e}")
        return None
    finally:
        conn.close()

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
        if any(error_word in line.lower() for error_word in ["warn", "error", "fatal"]):
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

# Main function to monitor logs and count "error" occurrences
def monitor_pods():
    # Get the namespaces from environment variable
    namespaces = os.getenv('NAMESPACE', 'default').split(",")
    
    while True:
        for namespace in namespaces:
            namespace = namespace.strip()  # Strip any extra whitespace
            print(f"Monitoring namespace: {namespace}")
            
            pods = v1.list_namespaced_pod(namespace)
            
            for pod in pods.items:
                pod_name = pod.metadata.name
                for container in pod.spec.containers:
                    container_name = container.name
                    current_time = datetime.datetime.now(timezone.utc)
                    
                    # Get the last stored timestamp from the database
                    last_stored_time = get_last_stored_time(namespace, pod_name, container_name)
                    
                    if last_stored_time:
                        since_time = last_stored_time
                    else:
                        # If no last stored time, fetch all logs
                        since_time = current_time - datetime.timedelta(days=365)  # 1 year, adjust if needed
                    
                    since_seconds = int((current_time - since_time).total_seconds())
                    
                    # Get logs since the last stored time (or all logs if no timestamp found)
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
