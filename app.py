import streamlit as st
import pandas as pd
import numpy as np
import duckdb
import time
import random
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# -------------------------------------------------
# App Config
# -------------------------------------------------
st.set_page_config(
    page_title="Enterprise AI Platform (Free Streamlit Demo)",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def init_state():
    if "auth" not in st.session_state:
        st.session_state.auth = False
        st.session_state.role = "User"
        st.session_state.user = None

    if "audit_logs" not in st.session_state:
        st.session_state.audit_logs = []  # list of dicts

    if "model_registry" not in st.session_state:
        st.session_state.model_registry = {}  # version -> dict(model, metrics, features)
        st.session_state.active_model_version = None

    if "etl_df" not in st.session_state:
        st.session_state.etl_df = None      # cleaned dataframe
        st.session_state.train_cache = {}   # store train/test splits for drift sim

def log_event(user, role, action, details=""):
    st.session_state.audit_logs.append({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "user": user or "guest",
        "role": role,
        "action": action,
        "details": details
    })

def guard_admin(pagename: str):
    admin_pages = ["CI/CD Pipeline", "Cloud Config (YAML)", "Multi-Cloud Load Balancing", "Audit Logs"]
    if st.session_state.role != "Admin" and pagename in admin_pages:
        st.error("🚫 Access denied – Admins only")
        st.stop()

init_state()

# -------------------------------------------------
# Fake Zero Trust Auth (Demo)
# -------------------------------------------------
if not st.session_state.auth:
    st.title("🔐 Login (Zero Trust Architecture) — Demo")
    with st.form("login"):
        user = st.text_input("Demo Username")
        pwd = st.text_input("Demo Password", type="password")
        role = st.selectbox("Role", ["User", "Admin"])
        submitted = st.form_submit_button("Login")
    if submitted:
        if user and pwd:
            st.session_state.auth = True
            st.session_state.role = role
            st.session_state.user = user
            st.success(f"Welcome {user} ({role})")
            log_event(user, role, "login", "User logged in")
            st.rerun()
        else:
            st.error("Enter any non-empty username & password for this demo.")
    st.stop()

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.title("🛰️ Cloud ML Microservices")
    st.caption("World’s Best Free Streamlit **enterprise AI platform demo** (concept showcase).")
    st.markdown(f"**User:** `{st.session_state.user}`  \n**Role:** `{st.session_state.role}`")

    page = st.radio("Go to", [
        "About / README",
        "Architecture Diagram",
        "ETL / Database",
        "Train & Deploy",
        "Monitoring & Drift",
        "CI/CD Pipeline",
        "Cloud Config (YAML)",
        "Multi-Cloud Load Balancing",
        "API-First Demo",
        "Agentic Research Mode",
        "Audit Logs",
    ])

    # Quick actions
    if st.button("🔒 Logout"):
        log_event(st.session_state.user, st.session_state.role, "logout", "User logged out")
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# -------------------------------------------------
# 0. About / README
# -------------------------------------------------
if page == "About / README":
    st.title("🛰️ Enterprise AI Platform — Free Streamlit Demo")
    st.markdown("""
This app is a **conceptual, end-to-end enterprise AI/Cloud/MLOps platform** demo running entirely on **free Streamlit Cloud**.

**What it showcases (USP/UVP):**
- 🔐 Zero Trust-style login + **RBAC** (Admin/User)
- 🗄️ **ETL** on CSV + in-memory **DuckDB** + **SQL runner**
- 🤖 Train & deploy demo (LogReg, RandomForest) + **Confusion Matrix** + metrics
- 📈 **Monitoring & Drift detection** (distribution shift vs training set)
- ⚙️ **CI/CD** pipeline simulator
- 📜 **Cloud YAML** + devops snippets
- 🌍 Multi-cloud **latency/cost** trade-offs
- 🔗 **API-first** approach example
- 🧑‍🔬 **Agentic research** orchestration (Planner → Researcher → Critic → Summarizer)
- 📜 **Audit logs** (who did what, when)

> This is a **showcase** — not a production system. It communicates **architecture thinking** and **2030-ready** skills on a lightweight stack.
    """)

# -------------------------------------------------
# 1. Architecture Diagram (pure Streamlit GraphViz)
# -------------------------------------------------
elif page == "Architecture Diagram":
    st.title("📊 Solution Architecture Design")
    st.markdown("**Data → ETL → DuckDB → Train → Deploy → Monitor → CI/CD → Multi-Cloud**")
    # Using st.graphviz_chart with DOT directly (no external binaries needed)
    dot = """
    digraph {
        rankdir=LR;
        node [shape=box, style=rounded];
        A[label="Data Upload"];
        B[label="Preprocess (ETL)"];
        C[label="DuckDB (In-Memory)"];
        D[label="Train Model"];
        E[label="Deploy API (Demo)"];
        F[label="Monitoring & Drift"];
        G[label="CI/CD Pipeline"];
        H[label="Multi-Cloud LB"];

        A -> B -> C -> D -> E -> F;
        D -> G;
        E -> H;
    }
    """
    st.graphviz_chart(dot)
    log_event(st.session_state.user, st.session_state.role, "view", "Architecture Diagram")

# -------------------------------------------------
# 2. ETL / Database with DuckDB & SQL Runner
# -------------------------------------------------
elif page == "ETL / Database":
    st.title("🗄️ ETL / ELT + DuckDB (In-Memory)")

    tabs = st.tabs(["Upload & Clean", "SQL Runner", "Data Profiling"])
    with tabs[0]:
        file = st.file_uploader("Upload CSV (<= 5MB recommended)", type=["csv"])
        if file:
            try:
                # read small CSV comfortably on free tier
                raw = pd.read_csv(file)
                st.subheader("📂 Raw Data")
                st.dataframe(raw.head(50), use_container_width=True)

                # Simple clean: drop rows with any NA
                df_clean = raw.dropna()
                st.subheader("✨ Cleaned Data (NA dropped)")
                st.dataframe(df_clean.head(50), use_container_width=True)

                # Save to session for other tabs/pages
                st.session_state.etl_df = df_clean

                # Load into DuckDB
                con = duckdb.connect(database=":memory:")
                con.register("df_clean", df_clean)
                con.execute("CREATE TABLE data AS SELECT * FROM df_clean")
                count = con.execute("SELECT COUNT(*) AS rows FROM data").fetchdf()
                st.success("✅ Data loaded into DuckDB (in-memory)")
                st.write(count)

                log_event(st.session_state.user, st.session_state.role, "etl_load",
                          f"Loaded CSV with {len(df_clean)} rows")
            except Exception as e:
                st.error(f"Failed to process CSV: {e}")

    with tabs[1]:
        st.subheader("🧪 SQL Runner (DuckDB)")
        if st.session_state.etl_df is None:
            st.info("Upload a CSV in **Upload & Clean** first.")
        else:
            con = duckdb.connect(database=":memory:")
            con.register("data", st.session_state.etl_df)
            default_sql = "SELECT * FROM data LIMIT 10"
            query = st.text_area("SQL", value=default_sql, height=120)
            if st.button("Run SQL"):
                try:
                    res = con.execute(query).fetchdf()
                    st.dataframe(res, use_container_width=True)
                    log_event(st.session_state.user, st.session_state.role, "sql_query", query)
                except Exception as e:
                    st.error(f"SQL error: {e}")

    with tabs[2]:
        st.subheader("📊 Data Profiling (Quick)")
        if st.session_state.etl_df is None:
            st.info("Upload a CSV first.")
        else:
            df = st.session_state.etl_df
            c1, c2 = st.columns(2)
            with c1:
                st.write("Shape:", df.shape)
                st.write("Columns:", list(df.columns))
                st.write("Dtypes:", df.dtypes.astype(str).to_dict())
            with c2:
                st.write("Describe:")
                st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

# -------------------------------------------------
# 3. Train & Deploy (Demo)
# -------------------------------------------------
elif page == "Train & Deploy":
    st.title("🤖 Train & Deploy ML Model (Demo)")
    log_event(st.session_state.user, st.session_state.role, "view", "Train & Deploy")

    # Dataset choice
    demo = st.selectbox("Dataset", ["Iris (sklearn)", "Synthetic Binary", "Use CSV from ETL"])
    if demo == "Iris (sklearn)":
        from sklearn.datasets import load_iris
        data = load_iris(as_frame=True)
        df = data.frame.copy()
        X, y = df.drop("target", axis=1), df["target"]
    elif demo == "Synthetic Binary":
        from sklearn.datasets import make_classification
        X_arr, y_arr = make_classification(n_samples=400, n_features=6, n_informative=4,
                                           n_redundant=0, random_state=42)
        X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(X_arr.shape[1])])
        y = pd.Series(y_arr, name="target")
        df = X.copy()
        df["target"] = y
    else:
        if st.session_state.etl_df is None:
            st.warning("No ETL dataframe available. Upload a CSV in **ETL / Database**.")
            st.stop()
        # Assume last column is target if found; else ask user to pick
        df = st.session_state.etl_df.copy()
        cols = list(df.columns)
        target_col = st.selectbox("Select target column", cols)
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        # Ensure numeric-only for simple demo models
        X = X.select_dtypes(include=[np.number])
        st.info("Using numeric columns only for training in this demo.")

    st.write("📂 Dataset preview")
    st.dataframe(df.head(20), use_container_width=True)

    model_name = st.selectbox("Model", ["Logistic Regression", "Random Forest"])
    test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05, help="Holdout fraction for test")

    if st.button("Train"):
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) < 20 else None
            )

            if model_name == "Logistic Regression":
                model = LogisticRegression(max_iter=300)
            else:
                model = RandomForestClassifier(n_estimators=200, random_state=42)

            with st.spinner("Training..."):
                time.sleep(0.5)
                model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{acc:.3f}")
            c2.metric("Train Size", len(X_train))
            c3.metric("Test Size", len(X_test))

            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            st.subheader("📋 Classification Report")
            st.json(report)

            # Confusion Matrix
            st.subheader("🧭 Confusion Matrix")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
            st.pyplot(fig)

            # Save into model registry
            version = f"v{len(st.session_state.model_registry) + 1}"
            st.session_state.model_registry[version] = {
                "model_name": model_name,
                "model": model,
                "metrics": {"accuracy": float(acc)},
                "features": list(X.columns),
                "X_train_mean": X_train.mean(numeric_only=True),  # for drift
                "X_test": X_test.head(50),  # small slice for quick demo prediction later
            }
            st.session_state.active_model_version = version
            st.success(f"✅ Trained & registered model as **{version}** (active).")
            log_event(st.session_state.user, st.session_state.role, "train",
                      f"Trained {model_name}, acc={acc:.3f}, version={version}")

            # Cache split for Monitoring page (drift calc)
            st.session_state.train_cache["X_train_mean"] = X_train.mean(numeric_only=True)
            st.session_state.train_cache["X_test"] = X_test.copy()
        except Exception as e:
            st.error(f"Training failed: {e}")

    st.subheader("🔮 Predict with Active Model")
    if not st.session_state.model_registry or not st.session_state.active_model_version:
        st.info("Train a model first.")
    else:
        versions = list(st.session_state.model_registry.keys())
        active = st.selectbox("Model Version", versions, index=versions.index(st.session_state.active_model_version))
        reg = st.session_state.model_registry[active]
        model = reg["model"]
        feats = reg["features"]

        st.caption(f"Model: **{reg['model_name']}** | Features: {feats}")
        sample_source = st.radio("Prediction Input", ["Use stored test sample", "Manual input"], horizontal=True)

        if sample_source == "Use stored test sample" and "X_test" in reg:
            sample = reg["X_test"].head(1)
            st.write(sample)
            pred = model.predict(sample)[0]
            st.success(f"Prediction: **{pred}**")
        else:
            inputs = {}
            cols = st.columns(min(4, len(feats)) or 1)
            for i, f in enumerate(feats):
                with cols[i % len(cols)]:
                    inputs[f] = st.number_input(f, value=0.0)
            if st.button("Predict"):
                row = pd.DataFrame([inputs])[feats]
                try:
                    pred = model.predict(row)[0]
                    st.success(f"Prediction: **{pred}**")
                    log_event(st.session_state.user, st.session_state.role, "predict",
                              f"version={active}, input={inputs}, pred={pred}")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

# -------------------------------------------------
# 4. Monitoring & Drift
# -------------------------------------------------
elif page == "Monitoring & Drift":
    st.title("📈 Monitoring & Drift Detection")
    log_event(st.session_state.user, st.session_state.role, "view", "Monitoring & Drift")

    if not st.session_state.model_registry or "X_train_mean" not in st.session_state.train_cache:
        st.info("Train a model in **Train & Deploy** first.")
        st.stop()

    # Compare training feature means vs. current batch means
    X_train_mean = st.session_state.train_cache["X_train_mean"]

    # Simulate a current batch: either use stored X_test or add light noise
    X_test_ref = st.session_state.train_cache.get("X_test")
    if X_test_ref is None:
        st.warning("No cached test set; please retrain a model.")
        st.stop()
    current_batch = X_test_ref.copy()
    # Optional perturbation slider to simulate drift
    noise = st.slider("Drift perturbation", 0.0, 1.0, 0.1, 0.05)
    current_batch = current_batch + np.random.normal(0, noise, current_batch.shape)

    train_means = X_train_mean.reindex(current_batch.columns).fillna(0)
    batch_means = current_batch.mean(numeric_only=True)

    drift_score = float(np.abs(train_means - batch_means).mean())
    st.metric("Drift Score (mean abs diff)", f"{drift_score:.4f}")
    if drift_score > 0.7:
        st.error("⚠️ Drift Detected! Recommend retraining.")
    else:
        st.success("✅ No major drift.")

    with st.expander("Details: training vs batch means"):
        comp = pd.DataFrame({"train_mean": train_means, "batch_mean": batch_means})
        comp["abs_diff"] = (comp["train_mean"] - comp["batch_mean"]).abs()
        st.dataframe(comp, use_container_width=True)

# -------------------------------------------------
# 5. CI/CD Pipeline (Admin)
# -------------------------------------------------
elif page == "CI/CD Pipeline":
    guard_admin(page)
    st.title("⚙️ CI/CD Pipeline Simulator")
    st.caption("Simulated GitHub Actions-style pipeline: Lint → Test → Build → Deploy")
    steps = ["Lint Code", "Run Tests", "Build Container", "Deploy to Cloud"]
    status = {}
    for s in steps:
        status[s] = st.checkbox(f"{s}", value=True)
    if st.button("Run Pipeline"):
        with st.spinner("Running pipeline..."):
            time.sleep(1.2)
        st.success("✅ Pipeline succeeded (simulated).")
        log_event(st.session_state.user, st.session_state.role, "cicd_run", f"Steps: {status}")

    st.subheader("Example workflow (YAML)")
    st.code("""\
name: CI-CD
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Install
      run: pip install -r requirements.txt
    - name: Lint
      run: flake8 .
    - name: Test
      run: pytest -q
    - name: Build & Deploy
      run: echo 'Deploying model... (simulated)'
""", language="yaml")

# -------------------------------------------------
# 6. Cloud Config (YAML) (Admin)
# -------------------------------------------------
elif page == "Cloud Config (YAML)":
    guard_admin(page)
    st.title("📜 Cloud Config Examples")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Kubernetes Deployment")
        st.code("""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-service
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ml
  template:
    metadata:
      labels:
        app: ml
    spec:
      containers:
      - name: ml-container
        image: ml:latest
        ports:
        - containerPort: 8080
""", language="yaml")

    with col2:
        st.caption("Dockerfile (for Streamlit app)")
        st.code("""\
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
""", language="dockerfile")

# -------------------------------------------------
# 7. Multi-Cloud Load Balancing (Admin)
# -------------------------------------------------
elif page == "Multi-Cloud Load Balancing":
    guard_admin(page)
    st.title("🌍 Multi-Cloud Load Balancing (Simulated)")
    cloud = st.radio("Select region", ["AWS - us-east-1", "GCP - europe-west1", "Azure - eastus"], horizontal=True)
    cfg = {
        "AWS - us-east-1": {"latency": random.randint(50, 120), "cost": 0.12},
        "GCP - europe-west1": {"latency": random.randint(100, 200), "cost": 0.10},
        "Azure - eastus": {"latency": random.randint(80, 150), "cost": 0.11},
    }[cloud]
    c1, c2 = st.columns(2)
    c1.metric("Latency", f"{cfg['latency']} ms")
    c2.metric("Cost / 1K req", f"${cfg['cost']:.2f}")

    st.info("Tip: Choose lower latency for user experience; choose lower cost for batch workloads.")
    log_event(st.session_state.user, st.session_state.role, "multicloud_check", f"{cloud} -> {cfg}")

# -------------------------------------------------
# 8. API-First Demo
# -------------------------------------------------
elif page == "API-First Demo":
    st.title("🔗 API-First Architecture (Example)")
    st.write("Example: cURL call to a model endpoint. Replace with your own URL.")
    st.code("""\
curl -X POST https://example.com/predict \\
  -H "Content-Type: application/json" \\
  -d '{"features":[1.2, 0.3, 5.1, 0.8]}'
""", language="bash")
    log_event(st.session_state.user, st.session_state.role, "view", "API-First Demo")

# -------------------------------------------------
# 9. Agentic Research Mode
# -------------------------------------------------
elif page == "Agentic Research Mode":
    st.title("🧑‍🔬 Agentic Research Mode (Simulated)")
    query = st.text_input("Research question", "Latest trends in AI Agents?")
    if st.button("Run Research Agents"):
        with st.spinner("Agents researching..."):
            time.sleep(1)
        msgs = [
            ("Planner", f"We will research: '{query}'. Subtasks: search, extract, summarize."),
            ("Researcher", "I found 3 mock papers: Multi-Agent AI 2025, Agent Orchestration, LLM Autonomy."),
            ("Critic", "Paper 2 has weak methodology. Prefer Paper 1 and 3."),
            ("Summarizer", "Key trend: orchestration, tool-use, and safety guardrails."),
            ("DecisionMaker", "Recommendation: prototype small multi-agent workflows with strict output checks.")
        ]
        for role, msg in msgs:
            st.chat_message(role).write(msg)
        log_event(st.session_state.user, st.session_state.role, "agents_run", query)

# -------------------------------------------------
# 10. Audit Logs (Admin)
# -------------------------------------------------
elif page == "Audit Logs":
    guard_admin(page)
    st.title("📜 Audit Logs")
    if len(st.session_state.audit_logs) == 0:
        st.info("No events yet.")
    else:
        st.dataframe(pd.DataFrame(st.session_state.audit_logs), use_container_width=True)
