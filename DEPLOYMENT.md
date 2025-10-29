# BDH Web Deployment

This directory contains the web deployment of the Baby Dragon Hatchling (BDH) model.

## 🚀 Quick Start (Local Development)

### 1. Install Dependencies

```bash
pip install -r requirements-web.txt
```

### 2. (Optional) Train the Model

If you want to use a trained model instead of random initialization:

```bash
python train.py
```

This will create a `bdh_model.pt` file that the web app will automatically load.

### 3. Run the Web Application

```bash
python app.py
```

Then open your browser to: **http://localhost:5000**

## 🌐 Features

- **Interactive Text Generation**: Enter any prompt and watch BDH generate text
- **Adjustable Parameters**: 
  - Temperature (0.1-2.0)
  - Top-K sampling (1-100)
  - Max tokens (10-500)
- **Advanced Configuration**: Modify model architecture (layers, embedding size, heads)
- **Example Prompts**: Quick-start with pre-configured examples
- **Real-time Generation**: See results instantly
- **Mobile Responsive**: Works on all devices

## 📝 API Endpoints

### GET `/api/info`
Get model information and device status

### POST `/api/generate`
Generate text from a prompt

**Request Body:**
```json
{
  "prompt": "Your text here",
  "max_new_tokens": 100,
  "temperature": 0.8,
  "top_k": 40,
  "config": {
    "n_layer": 6,
    "n_embd": 256,
    "n_head": 4,
    "mlp_internal_dim_multiplier": 128,
    "vocab_size": 256,
    "dropout": 0.1
  }
}
```

**Response:**
```json
{
  "success": true,
  "prompt": "Your text here",
  "output": "Generated text...",
  "generated_text": "New text only...",
  "config": {...}
}
```

### GET `/api/examples`
Get pre-configured example prompts

### GET `/api/train-status`
Check if a trained model is available

## 🐳 Docker Deployment

### Build the Docker image:

```bash
docker build -t bdh-web .
```

### Run the container:

```bash
docker run -p 5000:5000 bdh-web
```

Access at: **http://localhost:5000**

## ☁️ Cloud Deployment Options

### Heroku

1. Install Heroku CLI
2. Create a new app: `heroku create your-bdh-app`
3. Deploy: `git push heroku main`

### Railway

1. Connect your GitHub repository to Railway
2. Railway will auto-detect the Dockerfile
3. Deploy with one click

### DigitalOcean App Platform

1. Create a new app from GitHub repo
2. Select Dockerfile as build method
3. Deploy

### Render

1. Create new Web Service
2. Connect repository
3. Select Docker as environment
4. Deploy

## 📊 Performance Notes

- **CPU Mode**: Works but slower generation (5-10 seconds per request)
- **GPU Mode**: Much faster generation (1-2 seconds per request)
- **Memory**: ~500MB RAM for default configuration
- **Concurrent Users**: Use gunicorn with multiple workers for production

## 🔧 Configuration

### Environment Variables

- `FLASK_ENV`: Set to `production` for production deployment
- `PORT`: Override default port (default: 5000)
- `WORKERS`: Number of gunicorn workers (default: 1)

### Model Configuration

Edit the `DEFAULT_CONFIG` in `app.py` to change default model settings:

```python
DEFAULT_CONFIG = {
    "n_layer": 6,          # Number of transformer layers
    "n_embd": 256,         # Embedding dimension
    "n_head": 4,           # Number of attention heads
    "mlp_internal_dim_multiplier": 128,
    "vocab_size": 256,
    "dropout": 0.1
}
```

## 🛡️ Security Considerations

For production deployment:

1. **Rate Limiting**: Add rate limiting to prevent abuse
2. **Input Validation**: Already implemented but review for your use case
3. **CORS**: Configured for all origins - restrict for production
4. **HTTPS**: Use a reverse proxy (nginx) or cloud platform SSL
5. **Authentication**: Add if needed for private deployment

## 🎨 Customization

### Frontend
- Edit `templates/index.html` to customize UI
- Modify CSS in the `<style>` section
- Add new features in JavaScript

### Backend
- Modify `app.py` to add new endpoints
- Adjust model loading logic
- Add caching or database storage

## 📚 Learn More

- **Paper**: [The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain](https://doi.org/10.48550/arXiv.2509.26507)
- **GitHub**: [github.com/pathwaycom/bdh](https://github.com/pathwaycom/bdh)
- **Pathway**: [pathway.com](https://pathway.com)

## 🐛 Troubleshooting

**Issue**: "Import flask could not be resolved"
- **Solution**: Install web dependencies: `pip install -r requirements-web.txt`

**Issue**: Generation is slow
- **Solution**: Use GPU if available, or reduce `max_new_tokens`

**Issue**: Out of memory
- **Solution**: Reduce model size (n_layer, n_embd) or use smaller batch size

**Issue**: Port already in use
- **Solution**: Change port: `app.run(port=5001)`

## 📄 License

Copyright 2025 Pathway Technology, Inc.

See LICENSE.md for details.
