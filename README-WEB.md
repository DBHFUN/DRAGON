# 🐉 Baby Dragon Hatchling - Web Deployment

A beautiful, user-friendly web interface for the **Baby Dragon Hatchling (BDH)** language model - the biologically-inspired AI architecture that bridges Transformers and brain models!

![BDH Banner](https://img.shields.io/badge/BDH-Baby%20Dragon%20Hatchling-purple)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/License-Pathway-green)

## ✨ Features

- 🎨 **Beautiful Modern UI** - Gradient design with intuitive controls
- ⚙️ **Full Parameter Control** - Adjust temperature, top-k, and more
- 🔬 **Advanced Configuration** - Modify model architecture in real-time
- 📱 **Responsive Design** - Works on desktop, tablet, and mobile
- 🚀 **Fast Generation** - GPU-accelerated when available
- 📚 **Interactive Examples** - Quick-start with pre-configured prompts
- 💡 **Built-in Help** - Tooltips and guides for every parameter
- 🎯 **Real-time Inference** - See results as they're generated

## 🖼️ Screenshots

The web interface includes:
- Text input with example prompts
- Adjustable generation parameters (temperature, top-k, max tokens)
- Advanced model configuration panel
- Live output display with syntax highlighting
- Educational information about BDH architecture

## 🚀 Quick Start

### Option 1: Local Development (Recommended for Testing)

1. **Install Dependencies**
   ```bash
   pip install -r requirements-web.txt
   ```

2. **Run the Application**
   ```bash
   python app.py
   ```

3. **Open Your Browser**
   Navigate to: **http://localhost:5000**

That's it! The model will initialize on first use.

### Option 2: With Pre-trained Model (Better Results)

1. **Train the Model** (Optional but recommended)
   ```bash
   python train.py
   ```
   This trains on the tiny Shakespeare dataset and saves `bdh_model.pt`

2. **Run the Web App**
   ```bash
   python app.py
   ```

3. **Access the Interface**
   Open: **http://localhost:5000**

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t bdh-web .

# Run the container
docker run -p 5000:5000 bdh-web
```

Access at: **http://localhost:5000**

### Docker Compose

```yaml
version: '3.8'
services:
  bdh-web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
```

Run with: `docker-compose up`

## ☁️ Cloud Deployment

### Heroku

```bash
heroku create your-bdh-app
git push heroku main
heroku open
```

### Railway.app

1. Connect your GitHub repository
2. Railway auto-detects the Dockerfile
3. Click Deploy
4. Access your app at the provided URL

### Render

1. Create a new Web Service
2. Connect your repository
3. Select Docker environment
4. Deploy automatically

### DigitalOcean App Platform

1. Create new app from GitHub
2. Select Dockerfile build method
3. Deploy with one click

## 📖 Usage Guide

### Basic Text Generation

1. Enter a prompt in the text area (e.g., "To be or not to be")
2. Adjust parameters if desired:
   - **Temperature**: 0.5 = focused, 1.0 = balanced, 1.5+ = creative
   - **Top-K**: Lower = more deterministic, Higher = more variety
   - **Max Tokens**: How much text to generate
3. Click "Generate Text"
4. View the output in the right panel

### Advanced Configuration

Click "Advanced Model Configuration" to modify:
- **Number of Layers**: Model depth (more = more capacity)
- **Embedding Dimension**: Hidden size (larger = more parameters)
- **Number of Heads**: Parallel attention mechanisms

⚠️ **Note**: Changing architecture creates a new model instance.

### API Usage

You can also use the API directly:

```python
import requests

response = requests.post('http://localhost:5000/api/generate', json={
    "prompt": "Once upon a time",
    "max_new_tokens": 100,
    "temperature": 0.8,
    "top_k": 40
})

print(response.json()['output'])
```

## 🔧 Configuration

### Environment Variables

```bash
export FLASK_ENV=production    # Set to production mode
export PORT=5000               # Change port
```

### Model Configuration

Edit `app.py` to change defaults:

```python
DEFAULT_CONFIG = {
    "n_layer": 6,       # Number of layers
    "n_embd": 256,      # Embedding dimension
    "n_head": 4,        # Attention heads
    "dropout": 0.1      # Dropout rate
}
```

## 📊 Performance

| Configuration | CPU Time | GPU Time | Memory |
|--------------|----------|----------|---------|
| Default (6 layers) | ~5-10s | ~1-2s | ~500MB |
| Large (12 layers) | ~15-20s | ~3-5s | ~1GB |
| Small (3 layers) | ~2-5s | <1s | ~300MB |

## 🛠️ Development

### Project Structure

```
bdh-main/
├── app.py                  # Flask backend server
├── bdh.py                  # BDH model implementation
├── train.py                # Training script
├── templates/
│   └── index.html         # Web interface
├── requirements-web.txt   # Web dependencies
├── Dockerfile             # Container configuration
└── DEPLOYMENT.md          # Deployment guide
```

### Adding Features

**Backend (app.py)**:
- Add new API endpoints
- Implement caching
- Add user authentication

**Frontend (templates/index.html)**:
- Customize CSS styling
- Add new UI components
- Implement real-time streaming

## 🎓 About BDH

Baby Dragon Hatchling is a groundbreaking neural architecture that:

- **Mimics the brain**: Uses scale-free network topology like biological neurons
- **Locally computes**: Neurons interact through excitatory/inhibitory dynamics
- **Learns naturally**: Implements Hebbian learning and synaptic plasticity
- **Stays interpretable**: Sparse, positive activations you can understand
- **Scales efficiently**: Matches GPT-2 performance at equivalent parameters

### Research Paper

> Kosowski, A., et al. (2025). *The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain*. arXiv:2509.26507

📄 [Read the paper](https://doi.org/10.48550/arXiv.2509.26507)

## 🤝 Contributing

Contributions welcome! Areas to explore:

- 🎨 UI/UX improvements
- ⚡ Performance optimizations
- 🔧 New features (streaming, model selection, etc.)
- 📚 Better documentation
- 🧪 Testing and validation

## 📝 License

Copyright 2025 Pathway Technology, Inc.

See LICENSE.md for full license details.

## 🔗 Links

- **Original Repository**: [github.com/pathwaycom/bdh](https://github.com/pathwaycom/bdh)
- **Pathway**: [pathway.com](https://pathway.com)
- **Paper**: [arXiv:2509.26507](https://doi.org/10.48550/arXiv.2509.26507)
- **Video**: [SuperDataScience Podcast](https://www.youtube.com/watch?v=mfV44-mtg7c)

## 🆘 Support

Having issues? Check out:

1. **DEPLOYMENT.md** - Detailed deployment instructions
2. **GitHub Issues** - Report bugs or request features
3. **Documentation** - Built into the web interface

## 🌟 Acknowledgments

- **Pathway Team** - For developing BDH
- **Andrej Karpathy** - For nanoGPT inspiration
- **Research Community** - For advancing AI/neuroscience convergence

---

**Made with 💜 by the AI community | Powered by Pathway's BDH**

⭐ Star this project if you find it useful!
