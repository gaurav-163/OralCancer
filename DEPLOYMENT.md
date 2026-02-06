# 🚀 Deploying to Streamlit Cloud

This guide will help you deploy the Oral Cancer Detection AI application to Streamlit Cloud.

## Prerequisites

- A GitHub account
- A Streamlit Cloud account (sign up at https://streamlit.io/cloud)
- A Cohere API key (get one free at https://cohere.com)

## Step-by-Step Deployment

### 1. Push Your Code to GitHub

```bash
# Initialize git repository (if not already done)
git init

# Add all files
git add .

# Commit your changes
git commit -m "Initial commit - Oral Cancer Detection AI"

# Add your GitHub repository as remote
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git push -u origin main
```

**Important:** The `.gitignore` file is configured to exclude the large model file (*.h5). The app will automatically download it from Hugging Face on first run.

### 2. Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Click "New app"
3. Select your GitHub repository
4. Configure the deployment:
   - **Main file path:** `streamlit_app.py`
   - **Python version:** 3.11 (recommended) or 3.9-3.12
   - **Branch:** main

### 3. Add Secrets

In the Streamlit Cloud dashboard for your app:

1. Click on "Settings" (⚙️)
2. Go to "Secrets"
3. Add your secrets in TOML format:

```toml
COHERE_API_KEY = "your-actual-cohere-api-key-here"
```

### 4. Deploy!

Click "Deploy" and wait for your app to build and launch. The first deployment will take 5-10 minutes as it:
- Installs all dependencies
- Downloads the model from Hugging Face
- Initializes the application

## Important Files for Deployment

The following files are essential for Streamlit Cloud deployment:

- ✅ `streamlit_app.py` - Entry point (must be in root directory)
- ✅ `requirements.txt` - Python dependencies
- ✅ `packages.txt` - System dependencies (OpenCV needs libGL)
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `.gitignore` - Excludes large model files

## Configuration Files

### streamlit_app.py
The main entry point that Streamlit Cloud will run. It imports and executes the main application from the `app/` directory.

### requirements.txt
All Python dependencies. Key packages:
- `streamlit` - Web framework
- `tensorflow` - ML model inference
- `langchain` & `langgraph` - AI agent workflow
- `cohere` - LLM provider
- `opencv-python` - Image processing

### packages.txt
System-level dependencies for OpenCV:
```
libgl1-mesa-glx
libglib2.0-0
```

### .streamlit/config.toml
Streamlit configuration with theme settings and server options.

## Model Loading

The application uses the Hugging Face Hub to download the model automatically:

- **Repository:** `gauravvv7/Oralcancer`
- **File:** `oral-cancer-model.h5`
- **Size:** ~258MB
- **Download:** Automatic on first run

The model is cached after first download, so subsequent restarts are faster.

## Environment Variables / Secrets

Required secrets in Streamlit Cloud:

| Secret | Description | Required |
|--------|-------------|----------|
| `COHERE_API_KEY` | Cohere API key for LLM features | Yes |

To add secrets:
1. Go to your app dashboard on Streamlit Cloud
2. Click "Settings" → "Secrets"
3. Add secrets in TOML format (see `.streamlit/secrets.toml.template`)

## Troubleshooting

### App fails to start

**Check logs:** In Streamlit Cloud dashboard, click "Manage app" → "Logs"

Common issues:
- Missing `COHERE_API_KEY` secret
- Python version incompatibility (use 3.9-3.12)
- Insufficient resources (model is large)

### Model download fails

The app will retry downloading from Hugging Face. If it continues to fail:
- Check Hugging Face is accessible
- Verify the model repository exists: https://huggingface.co/gauravvv7/Oralcancer

### OpenCV errors

Make sure `packages.txt` is present with:
```
libgl1-mesa-glx
libglib2.0-0
```

### Memory issues

The model requires significant memory (~1GB+). Streamlit Cloud free tier should handle this, but if you experience issues:
- Consider using Streamlit Cloud's paid tier for more resources
- Optimize model loading (load only when needed)

## Local Development

To test locally before deploying:

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variable
export COHERE_API_KEY="your-key-here"

# Run the app
streamlit run streamlit_app.py
```

## Post-Deployment

After successful deployment:

1. **Test the app** - Upload sample images and verify predictions
2. **Monitor logs** - Check for any errors or warnings
3. **Share the URL** - Your app will be at `https://your-app-name.streamlit.app`
4. **Custom domain** (optional) - Configure in Streamlit Cloud settings

## Updates

To update your deployed app:

```bash
# Make changes to your code
git add .
git commit -m "Update description"
git push

# Streamlit Cloud will automatically redeploy
```

## Resource Limits

Streamlit Cloud free tier:
- **CPU:** 1 vCPU
- **Memory:** 1 GB RAM
- **Storage:** Ephemeral (resets on restart)
- **Build time:** 10 minutes max

For production use with high traffic, consider the paid tier.

## Support

- **Streamlit Docs:** https://docs.streamlit.io/streamlit-community-cloud
- **Community Forum:** https://discuss.streamlit.io/
- **GitHub Issues:** For app-specific bugs

---

**Ready to deploy?** Follow the steps above and your Oral Cancer Detection AI will be live in minutes! 🎉
