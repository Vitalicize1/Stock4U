# 🚀 Stock4U Deployment

This directory contains files for deploying Stock4U to Streamlit Cloud.

## 📁 Files

### **Core Files**
- `streamlit_app.py` - Standalone Streamlit application for cloud deployment
- `requirements_streamlit.txt` - Minimal dependencies for Streamlit Cloud
- `.streamlit/config.toml` - Streamlit configuration

### **Documentation**
- `STREAMLIT_DEPLOYMENT.md` - Step-by-step deployment guide
- `env.example` - Environment configuration template

## 🎯 Quick Deployment

1. **Push your code to GitHub**
2. **Go to [share.streamlit.io](https://share.streamlit.io)**
3. **Connect your repository**
4. **Set main file path to**: `deployment/streamlit_app.py`
5. **Deploy!**

## 🔧 Configuration

### **For Streamlit Cloud:**
- Use `requirements_streamlit.txt` (minimal dependencies)
- Use `streamlit_app.py` (standalone version)
- Copy `.streamlit/config.toml` to root directory

### **For Local Development:**
- Use `../requirements.txt` (full dependencies)
- Use `../dashboard.py` (full version with backend)

## 📖 Documentation

- See `STREAMLIT_DEPLOYMENT.md` for detailed deployment guide
- See `../docs/QUICK_START.md` for user setup guide
- See `../README.md` for project overview

---

**Deploy Stock4U to the cloud in minutes! 🚀📈**
