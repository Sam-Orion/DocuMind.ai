import streamlit as st
import pandas as pd
import json
import time
import requests
import io
import plotly.express as px
from typing import Dict, Any, List

# --- Configuration ---
API_BASE_URL = "http://localhost:8000/api/v1"
HEALTH_URL = "http://localhost:8000/health"

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styling ---
st.markdown("""
<style>
    .metric-card {
        padding: 15px;
        border-radius: 8px;
        background-color: #f0f2f6;
        border: 1px solid #dcdcdc;
    }
    .status-badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .status-success { background-color: #d4edda; color: #155724; }
    .status-processing { background-color: #fff3cd; color: #856404; }
    .status-failed { background-color: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# --- API Client ---
class APIClient:
    def is_healthy(self) -> bool:
        try:
            resp = requests.get(HEALTH_URL, timeout=2)
            return resp.status_code == 200
        except:
            return False

    def upload_document(self, file) -> Dict:
        try:
            files = {"file": (file.name, file.getvalue(), file.type)}
            resp = requests.post(f"{API_BASE_URL}/process", files=files)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_result(self, doc_id: str) -> Dict:
        try:
            resp = requests.get(f"{API_BASE_URL}/result/{doc_id}")
            if resp.status_code == 200:
                return resp.json().get("data", {})
            return {}
        except:
            return {}

    def get_recent_documents(self, limit: int = 10) -> List[Dict]:
        try:
            resp = requests.get(f"{API_BASE_URL}/documents?limit={limit}")
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                # Ensure data is always a list
                return data if isinstance(data, list) else []
            return []
        except:
            return []
            
    def submit_correction(self, doc_id: str, updates: Dict) -> bool:
        try:
            resp = requests.post(f"{API_BASE_URL}/correct/{doc_id}", json={"updates": updates})
            return resp.status_code == 200
        except:
            return False

client = APIClient()

# --- Session State ---
if 'processed_uploads' not in st.session_state:
    st.session_state.processed_uploads = {} # {filename: doc_id}

# --- Functions ---
def render_sidebar():
    st.sidebar.title("DocuMind AI")
    
    # Status
    healthy = client.is_healthy()
    status_icon = "🟢" if healthy else "🔴"
    status_text = "Online" if healthy else "Offline"
    st.sidebar.caption(f"API Status: {status_icon} {status_text}")
    
    st.sidebar.divider()
    
    # Upload
    uploaded_files = st.sidebar.file_uploader(
        "Upload Documents", 
        type=['pdf', 'jpg', 'png', 'jpeg'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.sidebar.button(f"Process {len(uploaded_files)} Files", type="primary"):
            progress_bar = st.sidebar.progress(0)
            for idx, file in enumerate(uploaded_files):
                if file.name not in st.session_state.processed_uploads:
                    res = client.upload_document(file)
                    if res.get("status") == "success":
                         st.session_state.processed_uploads[file.name] = res["data"]["document_id"]
                progress_bar.progress((idx + 1) / len(uploaded_files))
            st.rerun()

    # History
    st.sidebar.divider()
    st.sidebar.subheader("History")
    recent = client.get_recent_documents()
    for doc in recent:
        label = f"{doc.get('filename','unknown')} ({doc.get('status','?')})"
        if st.sidebar.button(label, key=doc.get('id')):
             st.session_state.selected_doc_id = doc.get('id')

def render_confidence_chart(flat_data):
    if not flat_data:
        return
    df = pd.DataFrame(flat_data)
    fig = px.bar(
        df, 
        x='Field', 
        y='Confidence', 
        color='Confidence',
        color_continuous_scale=['red', 'yellow', 'green'],
        range_y=[0, 1],
        title="Extraction Confidence"
    )
    st.plotly_chart(fig, use_container_width=True)

def render_main():
    # 1. Dashboard View (Processed Files Tracker)
    if not getattr(st.session_state, 'selected_doc_id', None):
        st.header("Dashboard")
        if st.session_state.processed_uploads:
            st.subheader("Current Batch")
            status_data = []
            
            # Simple polling simulation triggers on rerun
            # In a real app we might use st_autorefresh or similar
            
            for fname, doc_id in st.session_state.processed_uploads.items():
                res = client.get_result(doc_id)
                status = res.get("status", "processing")
                status_data.append({
                    "Filename": fname,
                    "Status": status,
                    "Type": res.get("document_type", "-"),
                    "ID": doc_id
                })
            
            df_status = pd.DataFrame(status_data)
            
            # Custom Rendering
            for _, row in df_status.iterrows():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
                col1.write(f"**{row['Filename']}**")
                
                s = row['Status']
                badge_class = f"status-{s}" if s in ['completed', 'failed'] else "status-processing"
                col2.markdown(f"<span class='status-badge {badge_class}'>{s.upper()}</span>", unsafe_allow_html=True)
                
                col3.write(row['Type'])
                if col4.button("View", key=f"view_{row['ID']}"):
                    st.session_state.selected_doc_id = row['ID']
                    st.rerun()
            
            if st.button("Refresh Status"):
                st.rerun()
        else:
             st.info("Upload documents via the sidebar to start.")

    # 2. Detail View
    else:
        doc_id = st.session_state.selected_doc_id
        if st.button("← Back to Dashboard"):
            del st.session_state.selected_doc_id
            st.rerun()
            
        data = client.get_result(doc_id)
        if not data:
            st.error("Could not load document details.")
            return

        st.title(f"📄 {data.get('filename')}")
        
        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Status", data.get("status").upper())
        m2.metric("Type", data.get("document_type", "Unknown"))
        conf = data.get("confidence")
        m3.metric("Confidence", f"{conf:.2%}" if conf else "N/A")
        
        extracted = data.get("extracted_data", {})
        
        # Flatten for Table
        flat_data = []
        # Key -> Value map for editor
        table_data = []
        
        for k, v in extracted.items():
            # Handle list of dicts structure (v is list of extractions)
            if isinstance(v, list) and len(v) > 0:
                 for item in v:
                     flat_data.append({
                         "Field": k,
                         "Value": item.get('value'),
                         "Confidence": item.get('confidence')
                     })
                     table_data.append({
                         "Field": k,
                         "Value": item.get('value')
                     })
            # Handle direct values for retro-compatibility
            elif isinstance(v, str):
                flat_data.append({"Field": k, "Value": v, "Confidence": 1.0})
                table_data.append({"Field": k, "Value": v})

        # Tabs
        tab1, tab2, tab3 = st.tabs(["Extraction & Correction", "Visuals", "Raw Data"])
        
        with tab1:
            st.subheader("Extracted Data")
            
            # Interactive Editor for Corrections
            if table_data:
                df_editor = pd.DataFrame(table_data)
                edited_df = st.data_editor(
                    df_editor,
                    num_rows="dynamic",
                    key=f"editor_{doc_id}",
                    use_container_width=True
                )
                
                if st.button("Save Corrections", type="primary"):
                    # Diff Logic (Simple: Send all non-empty fields)
                    updates = {row["Field"]: row["Value"] for _, row in edited_df.iterrows() if row["Field"]}
                    if client.submit_correction(doc_id, updates):
                        st.success("Corrections saved!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Failed to save corrections.")
            else:
                st.info("No data extracted yet.")

        with tab2:
            st.subheader("Confidence Scores")
            render_confidence_chart(flat_data)

        with tab3:
            st.json(data)

def main():
    render_sidebar()
    render_main()

if __name__ == "__main__":
    main()
