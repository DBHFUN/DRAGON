# 🚀 Quick Setup Guide

## Installation Steps

### 1. Install Dependencies

Open PowerShell or terminal in this directory and run:

**Windows (PowerShell):**
```powershell
pip install -r requirements-web.txt
```

**Linux/Mac:**
```bash
pip3 install -r requirements-web.txt
```

### 2. Run the Application

**Option A: Using the startup script (Recommended)**

Windows:
```powershell
.\start-web.ps1
```

Linux/Mac:
```bash
chmod +x start-web.sh
./start-web.sh
```

**Option B: Direct Python command**

```bash
python app.py
```

### 3. Access the Web Interface

Open your browser and navigate to:
```
http://localhost:5000
```

## 🎯 First-Time Use

The application works in two modes:

### Mode 1: Random Initialization (Immediate)
- Just run `python app.py`
- The model starts with random weights
- You can test the interface immediately
- Output will be random text

### Mode 2: Trained Model (Recommended)
1. First train the model:
   ```bash
   python train.py
   ```
   This takes 5-10 minutes and creates `bdh_model.pt`

2. Then run the web app:
   ```bash
   python app.py
   ```

3. Now the model generates more coherent Shakespeare-style text!

## 📱 Using the Interface

1. **Enter a Prompt**: Type any text to continue from
2. **Adjust Parameters**:
   - Temperature: 0.5 (focused) to 1.5 (creative)
   - Top-K: 10-50 for best results
   - Max Tokens: 50-200 recommended
3. **Click Generate**: Wait a few seconds
4. **View Output**: See your generated text!

## 🐛 Troubleshooting

**"Import flask could not be resolved"**
→ Run: `pip install -r requirements-web.txt`

**"Port 5000 already in use"**
→ Edit app.py, change last line to: `app.run(debug=True, host='0.0.0.0', port=5001)`

**Generation is slow**
→ Normal on CPU (5-10 seconds). Use GPU for faster results.

**Connection refused**
→ Make sure `python app.py` is running in the terminal

## 🎨 Customization

- **Change UI**: Edit `templates/index.html`
- **Modify Model**: Edit default config in `app.py`
- **Add Features**: See `DEPLOYMENT.md` for details

## 📚 More Information

- Full deployment guide: `DEPLOYMENT.md`
- Web-specific README: `README-WEB.md`
- Original project: `README.md`

## ✅ Quick Test

After starting the server, try this:

1. Go to http://localhost:5000
2. Click one of the example prompts
3. Click "Generate Text"
4. You should see output in a few seconds!

Enjoy experimenting with BDH! 🐉
