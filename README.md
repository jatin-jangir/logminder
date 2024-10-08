# logminder


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


