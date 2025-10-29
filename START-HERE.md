# 🚀 START HERE - BDH Chat Interface

## What You're Looking At

This is a **ChatGPT-style chat interface** for Baby Dragon Hatchling (BDH) - an AI that works like your brain! 🧠

## ⚡ Quick Start (3 Commands)

### 1️⃣ Install
```bash
pip install -r requirements-web.txt
```

### 2️⃣ Train (THIS IS REQUIRED!)
```bash
python train.py
```
⏱️ Takes 5-10 minutes. **Don't skip this!**

### 3️⃣ Chat
```bash
python app.py
```
Then open: **http://localhost:5000**

## ❓ Why Do I Need to Train?

**Without training:**
```
You: "Hello"
BDH: "\xc4\xcb\xec*O\xf1" ← Random garbage
```

**After training:**
```
You: "To be or not to be"
BDH: "To be or not to be, that is the question..." ← Real text!
```

The model starts with random "knowledge" and needs to learn from Shakespeare first!

## 🎯 What You'll Get

- **Chat Interface**: Talk to BDH like ChatGPT
- **Smart Warnings**: See if model needs training
- **Example Prompts**: Quick-start buttons
- **Settings Control**: Adjust creativity, length, etc.
- **Mobile Friendly**: Works on phone/tablet/desktop

## 📱 The Two Interfaces

1. **Chat Mode** (Default)
   - `http://localhost:5000`
   - Easy, conversational
   - **Start here!**

2. **Advanced Mode**
   - `http://localhost:5000/original`
   - Full technical controls
   - For researchers

## 🐛 Troubleshooting

**Seeing random characters?**
→ Run `python train.py` and wait for it to finish

**"No module named flask"?**
→ Run `pip install -r requirements-web.txt`

**Slow generation?**
→ Normal on CPU! Takes 5-10 seconds per response

## 📚 More Help

- **CHAT-GUIDE.md** - Detailed user guide
- **SETUP.md** - Setup instructions
- **DEPLOYMENT.md** - How to deploy online

## 🎉 That's It!

You're ready to chat with a brain-inspired AI!

---

**Questions? Check the other docs or open an issue on GitHub!**
