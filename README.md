# logminder - Log Monitor for Kubernetes Pods

Logminder monitors log files from Kubernetes pods, extracts important metrics related to log errors, and stores these metrics in a PostgreSQL database. It helps in identifying anomalies in log files and tracking key events within the Kubernetes environment.


## Prerequisites

- Python 3.x
- Kubernetes Python client (kubernetes) update the config file with the your kubernetes config file
- PostgreSQL Python client (psycopg2) 
- Access to Kubernetes cluster with appropriate permissions
- A running PostgreSQL instance with a database named logminder


## Installation

 #### Install Python dependencies using pip:
```bash
pip install kubernetes psycopg2
```

 #### Set the environment variables required by the script:
- POSTGRES_PASSWORD: PostgreSQL password for the postgres user.
- NAMESPACE: Kubernetes namespaces to monitor (comma-separated).
```bash
export POSTGRES_PASSWORD=$(kubectl get secret --namespace default my-postgresql -o jsonpath="{.data.postgres-password}" | base64 -d)
export NAMESPACE='test-ns-1,test-ns-2,test-ns-3'
```
 ## Configuration

 #### PostgreSQL Setup

Ensure you have a PostgreSQL database set up with the following table:
```sql
CREATE TABLE log_metrics (
    timestamp TIMESTAMP WITH TIME ZONE,
    namespace VARCHAR(255),
    pod VARCHAR(255),
    container VARCHAR(255),
    error_count INTEGER,
    non_error_count INTEGER
);
```
 #### Kubernetes Configuration
- The script tries to load an in-cluster configuration first.
- If not running inside a cluster, it will fall back to using your local kubeconfig file.

## How It Works

1. The script initializes a Kubernetes API client and connects to a PostgreSQL database.
2. It continuously monitors the specified Kubernetes namespaces.
3. For each pod and container, it fetches logs starting from the last timestamp stored in the PostgreSQL database.
4. Logs are parsed to count lines containing keywords like "warn", "error", or "fatal".
5. The error and non-error counts are then stored in the PostgreSQL database.
6. The script sleeps for 60 seconds before repeating the process.


 #### important commands
```
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

helm install my-postgresql bitnami/postgresql \
  --set auth.postgresPassword=my-password \
  --set primary.persistence.size=1Gi



export POSTGRES_PASSWORD=$(kubectl get secret --namespace default my-postgresql -o jsonpath="{.data.postgres-password}" | base64 -d)


kubectl port-forward --namespace default svc/my-postgresql 5432:5432 &
    PGPASSWORD="$POSTGRES_PASSWORD" psql --host 127.0.0.1 -U postgres -d postgres -p 5432

jatinjangir@Jatins-MacBook-Air ~ % psql -h 127.0.0.1 -p 5432 -U postgres


logminder=# CREATE TABLE log_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    namespace TEXT,
    pod TEXT,
    container TEXT,
    error_count INT,
    non_error_count INT
);
CREATE TABLE
logminder=# CREATE INDEX idx_log_metrics_timestamp ON log_metrics (timestamp);
CREATE INDEX
logminder=# CREATE INDEX idx_log_metrics_namespace_pod_container ON log_metrics (namespace, pod, container);
CREATE INDEX

```


