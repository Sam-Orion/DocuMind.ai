import streamlit as st
import pandas as pd
import json
import time
import io
from typing import Dict, Any

from src.pipeline import DocumentProcessor

# Page Config
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
        background-color: #f9f9f9;
        margin-bottom: 10px;
    }
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Processor (Cached)
@st.cache_resource
def get_processor():
    return DocumentProcessor()

processor = get_processor()

def color_confidence(val):
    """
    Color confidence scores: Green > 0.9, Yellow 0.7-0.9, Red < 0.7
    """
    if isinstance(val, (int, float)):
        if val > 0.9:
            color = '#d4edda' # Green
        elif val > 0.7:
            color = '#fff3cd' # Yellow
        else:
            color = '#f8d7da' # Red
        return f'background-color: {color}; color: black'
    return ''

def main():
    st.title("📄 DocuMind AI")
    st.markdown("### Intelligent Document Processing System")

    # --- Sidebar ---
    st.sidebar.header("Upload Document")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a file", 
        type=['png', 'jpg', 'jpeg', 'pdf'],
        help="Max file size: 10MB"
    )

    doc_type_option = st.sidebar.selectbox(
        "Document Type",
        ["Auto-detect", "Invoice", "Receipt", "Resume", "ID Document", "Business Card"]
    )

    process_btn = st.sidebar.button("Process Document", type="primary")

    # --- Main Content ---
    if uploaded_file is not None:
        # File details
        file_details = {
            "Filename": uploaded_file.name,
            "FileType": uploaded_file.type,
            "FileSize": f"{uploaded_file.size / 1024:.2f} KB"
        }
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.info(f"File: {uploaded_file.name}")
            # Preview (Image only for now, PDF preview requires extra handling)
            if uploaded_file.type.startswith('image'):
                st.image(uploaded_file, caption='Document Preview', use_column_width=True)
            else:
                st.write("PDF Preview not supported yet.")

        if process_btn:
            with col2:
                with st.spinner("Processing document... Please wait"):
                    # Read file bytes
                    bytes_data = uploaded_file.getvalue()
                    
                    # Run Pipeline
                    try:
                        results = processor.process_document(bytes_data)
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
                        return

                    if results.get("status") == "error":
                        st.error(f"Pipeline Error: {results.get('error')}")
                        return

                    # -- Display Results --
                    
                    # 1. Classification
                    st.divider()
                    st.subheader("Classification")
                    
                    detected_type = results.get("document_type", "Unknown")
                    confidence = results.get("confidence", 0.0)
                    
                    # Handle manual override logic if needed, currently just showing detected
                    final_type = detected_type if doc_type_option == "Auto-detect" else doc_type_option
                    
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        st.metric("Document Type", final_type)
                    with metric_col2:
                         st.metric("Confidence", f"{confidence:.2%}")
                    with metric_col3:
                         st.metric("Processing Time", f"{results['performance']['total_time']:.2f}s")
                    
                    if confidence > 0.8:
                        st.success(f"Confident it's a {final_type}")
                    elif confidence > 0.5:
                        st.warning(f"Likely a {final_type}")
                    else:
                        st.error(f"Unsure, detected {final_type}")

                    # 2. Extracted Fields
                    st.divider()
                    st.subheader("Extracted Data")
                    
                    extracted_fields = results.get("extracted_fields", {})
                    flat_data = []
                    
                    # Flatten the dictionary for table display
                    for field_type, items in extracted_fields.items():
                        for item in items:
                            flat_data.append({
                                "Field": field_type,
                                "Value": item['value'],
                                "Confidence": item['confidence'],
                                "Original Text": item.get('original_text', '')
                            })
                    
                    if flat_data:
                        df = pd.DataFrame(flat_data)
                        
                        # Apply coloring
                        st.dataframe(
                            df.style.map(color_confidence, subset=['Confidence'])
                                    .format({"Confidence": "{:.2%}"}),
                            use_container_width=True
                        )
                        
                        # Exports
                        csv = df.to_csv(index=False).encode('utf-8')
                        json_str = json.dumps(results, indent=2)
                        
                        btn_col1, btn_col2 = st.columns(2)
                        with btn_col1:
                            st.download_button(
                                "Download CSV",
                                csv,
                                "extracted_data.csv",
                                "text/csv",
                                key='download-csv'
                            )
                        with btn_col2:
                            st.download_button(
                                "Download JSON",
                                json_str,
                                "full_results.json",
                                "application/json",
                                key='download-json'
                            )

                    else:
                        st.info("No specific fields extracted.")

                    # 3. OCR Text
                    with st.expander("View Raw OCR Text"):
                        st.text_area("Full Text", results.get("text_content", ""), height=300)
                    
                    # 4. Debug Info
                    with st.expander("Debug & Performance Details"):
                        st.json(results.get("performance"))
                        st.json(results.get("classification_details"))

    else:
        st.info("Please upload a document to begin.")

if __name__ == "__main__":
    main()
