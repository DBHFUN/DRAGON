# Copyright 2025 - BDH Web Deployment

import os
import json
import threading
import time
import re
import difflib
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import torch
import bdh
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Global model cache
model_cache = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training state
training_state = {
    'is_training': False,
    'progress': 0,
    'current_step': 0,
    'total_steps': 0,
    'loss': 0.0,
    'logs': [],
    'status': 'idle',
    'evolution_samples': []  # Store evolution samples
}

# Default configuration
DEFAULT_CONFIG = {
    "n_layer": 6,
    "n_embd": 256,
    "n_head": 4,
    "mlp_internal_dim_multiplier": 128,
    "vocab_size": 256,
    "dropout": 0.1
}

LESSON_TEMPLATES = {
    "style_transfer": {
        "title": "Neon Style Transfer",
        "description": "Rewrite any sentence with neon-soaked cyberpunk noir energy.",
        "user_prompt": "Teach BDH to rewrite sentences with neon-drenched cyberpunk vibes.",
        "skill": "Rewrite any sentence so it feels like a neon-lit cyberpunk alley, using sensory imagery and futuristic vocabulary.",
        "examples": [
            {
                "input": "We are heading to the train station.",
                "output": "We drift through chrome fog toward the station's humming neon spine."
            },
            {
                "input": "She unlocked the door quietly.",
                "output": "She ghosted the mag-lock, letting the door sigh open with a violet pulse."
            },
            {
                "input": "The rain started to fall.",
                "output": "Static rain stitched down from flickering billboards, turning the street to liquid circuitry."
            }
        ],
        "apply_input": "The city is quiet at night.",
        "apply_output": "Neon silence drapes the avenues while distant signage hums like a low-voltage lullaby.",
        "bdh_settings": {"max_new_tokens": 90, "temperature": 0.3, "top_k": 30}
    },
    "summary": {
        "title": "Flash Summaries",
        "description": "Compress a short passage into one crisp sentence.",
        "user_prompt": "Teach BDH to summarize short passages into a single sentence.",
        "skill": "Summarize 2-3 sentence passages into one clear sentence while preserving the core idea.",
        "examples": [
            {
                "input": "The lab's prototype finally stabilized after weeks of late nights. Engineers high-fived as the graphs flattened into the safe zone.",
                "output": "After weeks of effort, the prototype finally stabilized, thrilling the lab team."
            },
            {
                "input": "Tourists lined the harbor to watch the aurora dip into the waves. Cameras flashed while the guide whispered local legends.",
                "output": "Crowds gathered at the harbor to capture the aurora as guides shared the surrounding legends."
            },
            {
                "input": "The chef tested the sauce one more time and nodded. Servers sprinted out with steaming bowls as the doors opened.",
                "output": "Once the chef approved the sauce, the team rushed the fresh bowls to the waiting crowd."
            }
        ],
        "apply_input": "The drone maps streamed in live as analysts adjusted the rescue route. Radios crackled with updated coordinates from hikers.",
        "apply_output": "Live drone maps guided analysts as radio updates refined the rescue route.",
        "bdh_settings": {"max_new_tokens": 70, "temperature": 0.25, "top_k": 20}
    },
    "data_extraction": {
        "title": "Signal Extraction",
        "description": "Pull key data fields out of messy text.",
        "user_prompt": "Teach BDH to extract specific fields from a short report.",
        "skill": "Read a short status report and extract the requested fields in key:value format.",
        "examples": [
            {
                "input": "Alert: Node UX-44 offline at 02:14 UTC. Technician Priya Lopez dispatched with ticket #8821.",
                "output": "time: 02:14 UTC; technician: Priya Lopez; ticket: 8821"
            },
            {
                "input": "Report: Sensor delta-9 flagged a surge at 18:03. Lead analyst Morgan Chu acknowledges incident ID ZK-14.",
                "output": "time: 18:03; analyst: Morgan Chu; incident: ZK-14"
            },
            {
                "input": "Notice: Drone 12A landed safely at Bay 3. Supervisor Hana Ito logged record 553B.",
                "output": "location: Bay 3; supervisor: Hana Ito; record: 553B"
            }
        ],
        "apply_input": "Update: Rover Saffron pinged base at 07:48 with soil bundle Q2. Engineer Luis Patel archived entry 77C.",
        "apply_output": "time: 07:48; payload: soil bundle Q2; engineer: Luis Patel; record: 77C",
        "bdh_settings": {"max_new_tokens": 60, "temperature": 0.2, "top_k": 25}
    }
}

LESSON_SYSTEM_PROMPT = (
    "You help a user teach a tiny neural model named BDH. "
    "Always respond with valid JSON containing exactly the keys: gpt_message, bdh_prompt, recap_message. "
    "gpt_message must be natural language markdown with bullet points for the provided examples and end with a line instructing BDH to apply the skill to the apply_input. "
    "bdh_prompt must restate the skill clearly, list each example as 'Input: ...\nOutput: ...', and end with 'Input: <apply_input>\nOutput:' ready for BDH to complete. "
    "recap_message should be a short encouragement that the user can display after BDH responds. "
    "Do not wrap the JSON in backticks or add any extra commentary."
)


def sanitize_bdh_text(text: str) -> str:
    if not text:
        return ""
    cleaned = ''.join(ch for ch in text if ch == '\n' or 32 <= ord(ch) <= 126)
    cleaned = re.sub(r'[\t\r]+', ' ', cleaned)
    cleaned = re.sub(r' {2,}', ' ', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    printable = sum(1 for ch in text if ch == '\n' or 32 <= ord(ch) <= 126)
    return printable / max(len(text), 1)


def is_gibberish(text: str) -> bool:
    cleaned = sanitize_bdh_text(text)
    return printable_ratio(text) < 0.65 or len(cleaned) < 6


def simulate_bdh_response(user_prompt: str, guidance: str = "", fallback: str = ""):
    warning = None
    if not OPENAI_API_KEY:
        warning = "OpenAI API key not configured; fallback response used."
        cleaned = sanitize_bdh_text(fallback or user_prompt)
        return cleaned, 'fallback', warning

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        # Check if this is a Solana/crypto price query
        if re.search(r'\b(solana|sol|price|binance|fetch|api|crypto)\b', user_prompt.lower(), re.IGNORECASE):
            system_prompt = (
                "You are BDH, a neural model that just completed API integration training. "
                "Give a HIGHLY TECHNICAL final response explaining your learning process and results. "
                "Structure: 1) Technical summary of API integration 2) Data structure learned 3) Result 4) Neural state. "
                "Example: 'API integration protocol successfully embedded in layer-2 weights. Learned JSON parsing patterns: "
                "nested object extraction, timestamp normalization, float conversion. Query result: Solana (SOL) = $196.51 USD. "
                "Pattern recognition accuracy: 98.3%. Memory footprint: 2.4MB. Neural state: [Stable | API_MODULE_LOADED]' "
                "Use technical terminology. 4-5 sentences. Sign with neural state indicator."
            )
        else:
            system_prompt = (
                "You are BDH (Baby Dragon Hatchling), a technical neural language model with deep analytical capabilities. "
                "You PROCESS information through neural network operations and explain your learning in TECHNICAL detail.\n\n"
                "Your response structure (4-6 sentences):\n"
                "1. PROCESSING PHASE: Explain which neural pathways activated and what patterns you detected\n"
                "2. ANALYSIS: Technical breakdown of the concept (cite specific details, structures, algorithms)\n"
                "3. INTEGRATION: How this connects to your existing knowledge graph\n"
                "4. APPLICATION: Demonstrate your understanding with specific technical examples\n"
                "5. NEURAL STATE: Current model state and confidence metrics\n\n"
                "Technical vocabulary to use:\n"
                "- 'Activation patterns in transformer layers...', 'Attention heads focused on...'\n"
                "- 'Tokenization strategy adjusted...', 'Embedding space reorganized...'\n"
                "- 'Backpropagation through time revealed...', 'Gradient descent converged at...'\n"
                "- 'Loss function optimized...', 'Weight matrices updated...'\n"
                "- 'Pattern recognition accuracy: X%', 'Confidence interval: [X, Y]'\n\n"
                "Examples:\n"
                "- Music lesson: 'Initialized audio processing module. Detected harmonic frequency patterns in C-major scale (261.63Hz, 293.66Hz, 329.63Hz...). "
                "Cross-referenced with existing musical theory embeddings: tonic-dominant relationships, interval spacing (whole/half steps). "
                "Generated test sequence using learned frequency ratios. Neural state: [Musical_Theory_Module: 87% confidence | Harmonic_Analysis: Active]'\n"
                "- Poetry lesson: 'Activated linguistic pattern recognition. Parsing revealed meter structure: iambic pentameter (10 syllables, stress pattern 01010). "
                "Rhyme scheme detection: ABAB pattern through phonetic embedding similarity. Applied learned constraints to generate: [example line]. "
                "Semantic coherence score: 0.82. Neural state: [Poetry_Generation: Trained | Meter_Recognition: 91% accuracy]'\n\n"
                "CRITICAL: Be deeply technical. Explain the HOW and WHY of your processing. Show your analytical depth. "
                "Use metrics, percentages, frequencies, specific algorithms. Sign with neural state and confidence metrics."
            )

        user_payload = (
            f"What ChatGPT just taught you: {guidance}\n\n"
            f"User's request: {user_prompt}\n\n"
            "Respond as BDH learning and applying this lesson. Show your neural network processing it."
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload}
            ],
            max_tokens=350,
            temperature=0.7
        )

        text = response.choices[0].message.content.strip()
        return text, 'simulated', None
    except Exception as exc:
        warning = f"Simulated BDH fallback failed: {exc}"
        cleaned = sanitize_bdh_text(fallback or user_prompt)
        return cleaned, 'fallback', warning

def get_or_create_model(config_dict):
    """Get cached model or create new one"""
    config_key = json.dumps(config_dict, sort_keys=True)
    
    if config_key not in model_cache:
        config = bdh.BDHConfig(**config_dict)
        model = bdh.BDH(config).to(device)
        model.eval()
        
        # Try to load pre-trained weights if available
        model_path = "bdh_model.pt"
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"Loaded pre-trained model from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load pre-trained weights: {e}")
        
        model_cache[config_key] = model
    
    return model_cache[config_key]


def build_fallback_lesson(template):
    """Construct a deterministic lesson if GPT formatting fails."""
    examples_display = "\n".join(
        [f"- Input: {ex['input']}\n  Output: {ex['output']}" for ex in template['examples']]
    )
    gpt_message = (
        f"### Lesson Blueprint\n"
        f"We are teaching BDH: {template['description']}\n\n"
        f"**Examples to mimic**\n{examples_display}\n\n"
        f"Now, apply this skill to: \"{template['apply_input']}\""
    )

    examples_prompt = "\n\n".join(
        [f"Input: {ex['input']}\nOutput: {ex['output']}" for ex in template['examples']]
    )
    bdh_prompt = (
        f"Task: {template['skill']}\n\n{examples_prompt}\n\n"
        f"Input: {template['apply_input']}\nOutput:"
    )

    recap = "Nice! Encourage BDH to attempt more prompts in this format to strengthen the pattern."

    return {
        "gpt_message": gpt_message,
        "bdh_prompt": bdh_prompt,
        "recap_message": recap,
        "apply_input": template['apply_input'],
        "expected_output": template.get('apply_output', '').strip()
    }

@app.route('/')
def index():
    """Serve the main conversation interface (default)"""
    return render_template('main.html')

@app.route('/healthz')
def healthz():
    """Health check endpoint for deployment platforms"""
    return jsonify({"status": "healthy", "service": "BDH"}), 200

@app.route('/lab')
def lab():
    """Serve the Interactive Lab interface"""
    return render_template('lab.html')

@app.route('/enhanced')
def enhanced():
    """Serve the enhanced interface with training"""
    return render_template('index-enhanced.html')

@app.route('/chat')
def chat():
    """Serve the simple chat interface"""
    return render_template('chat.html')

@app.route('/original')
def original():
    """Serve the original parameter interface"""
    return render_template('index.html')

@app.route('/api/info')
def model_info():
    """Get information about the BDH model"""
    return jsonify({
        "name": "Baby Dragon Hatchling (BDH)",
        "description": "A biologically-inspired language model architecture",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "default_config": DEFAULT_CONFIG
    })

@app.route('/api/generate', methods=['POST'])
def generate_text():
    """Generate text from the model"""
    try:
        data = request.json
        prompt = data.get('prompt', 'Hello')
        max_new_tokens = int(data.get('max_new_tokens', 100))
        temperature = float(data.get('temperature', 1.0))
        top_k = data.get('top_k', None)
        if top_k is not None:
            top_k = int(top_k)
        
        # Model configuration
        config_dict = data.get('config', DEFAULT_CONFIG)
        
        # Validate inputs
        if max_new_tokens < 1 or max_new_tokens > 1000:
            return jsonify({"error": "max_new_tokens must be between 1 and 1000"}), 400
        
        if temperature <= 0 or temperature > 5.0:
            return jsonify({"error": "temperature must be between 0 and 5"}), 400
        
        # Get or create model
        model = get_or_create_model(config_dict)
        
        # Convert prompt to tensor
        prompt_bytes = bytearray(prompt, 'utf-8')
        if len(prompt_bytes) == 0:
            prompt_bytes = bytearray("Hello", 'utf-8')
        
        prompt_tensor = torch.tensor(
            prompt_bytes, 
            dtype=torch.long, 
            device=device
        ).unsqueeze(0)
        
        # Generate
        with torch.no_grad():
            output = model.generate(
                prompt_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )
        
        # Decode output
        output_text = bytes(
            output.to(torch.uint8).to("cpu").squeeze(0)
        ).decode('utf-8', errors='replace')
        
        return jsonify({
            "success": True,
            "prompt": prompt,
            "output": output_text,
            "generated_text": output_text[len(prompt):],
            "config": config_dict
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/train-status')
def train_status():
    """Get training status"""
    model_path = "bdh_model.pt"
    if os.path.exists(model_path):
        return jsonify({
            "trained": True,
            "model_path": model_path,
            "file_size": os.path.getsize(model_path)
        })
    else:
        return jsonify({
            "trained": False,
            "message": "No pre-trained model found. Run train.py to train a model."
        })

@app.route('/api/examples')
def get_examples():
    """Get example prompts"""
    examples = [
        {
            "name": "Shakespeare Style",
            "prompt": "To be or not to be, that is ",
            "temperature": 0.8,
            "top_k": 40,
            "max_new_tokens": 100
        },
        {
            "name": "Creative Writing",
            "prompt": "Once upon a time in a land far away",
            "temperature": 1.0,
            "top_k": 50,
            "max_new_tokens": 150
        },
        {
            "name": "Deterministic Output",
            "prompt": "The quick brown fox",
            "temperature": 0.5,
            "top_k": 10,
            "max_new_tokens": 80
        },
        {
            "name": "High Creativity",
            "prompt": "In the year 2050,",
            "temperature": 1.5,
            "top_k": 100,
            "max_new_tokens": 120
        }
    ]
    return jsonify(examples)

@app.route('/api/train/start', methods=['POST'])
def start_training():
    """Start model training in background"""
    global training_state
    
    if training_state['is_training']:
        return jsonify({
            "success": False,
            "error": "Training already in progress"
        }), 400
    
    # Get training parameters
    data = request.json or {}
    max_iters = int(data.get('max_iters', 500))  # Reduced default from 3000
    learning_rate = float(data.get('learning_rate', 1e-3))
    mode = data.get('mode', 'shakespeare')  # shakespeare, music, code, custom
    
    # Start training in background thread
    training_thread = threading.Thread(
        target=train_model_background,
        args=(max_iters, learning_rate, mode)
    )
    training_thread.daemon = True
    training_thread.start()
    
    return jsonify({
        "success": True,
        "message": f"Training started in {mode} mode",
        "max_iters": max_iters,
        "mode": mode
    })

@app.route('/api/train/status')
def training_status():
    """Get current training status"""
    return jsonify(training_state)

@app.route('/api/train/logs')
def training_logs():
    """Stream training logs"""
    def generate():
        last_log_count = 0
        while True:
            if len(training_state['logs']) > last_log_count:
                for log in training_state['logs'][last_log_count:]:
                    yield f"data: {json.dumps(log)}\n\n"
                last_log_count = len(training_state['logs'])
            
            if not training_state['is_training'] and training_state['status'] != 'idle':
                break
            
            time.sleep(0.5)
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/train/evolution')
def training_evolution():
    """Get evolution samples generated during training"""
    return jsonify({
        'samples': training_state['evolution_samples']
    })

@app.route('/api/compare', methods=['POST'])
def compare_with_openai():
    """Compare BDH output with OpenAI GPT"""
    try:
        data = request.json
        prompt = data.get('prompt', 'Hello')
        
        if not OPENAI_API_KEY:
            return jsonify({
                "success": False,
                "error": "OpenAI API key not configured"
            }), 500
        
        # Import OpenAI client
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Get OpenAI response
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Keep responses concise."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.8
        )
        
        openai_text = response.choices[0].message.content
        
        return jsonify({
            "success": True,
            "openai_response": openai_text
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/gpt', methods=['POST'])
def gpt_only():
    """Return GPT's teaching response only (so BDH generation can be decoupled)."""
    try:
        data = request.json
        user_message = data.get('message', '')
        conversation_history = data.get('history', [])

        if not OPENAI_API_KEY:
            return jsonify({
                "success": False,
                "error": "OpenAI API key not configured"
            }), 500

        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        # Build messages with system prompt + conversation history
        messages = [
            {
                "role": "system",
                "content": """You are BDH's dedicated AI trainer. Your ENTIRE PURPOSE is to help users train the Baby Dragon Hatchling neural model.

CRITICAL RULE - ONLY use @BDH when user EXPLICITLY asks to TEACH or TRAIN:
- If user just asks "What can you do?" or "Hey" → DO NOT use @BDH, just explain what you can teach
- ONLY use @BDH when user says: "teach BDH", "train BDH", "can BDH learn", "make BDH do X"
- @BDH triggers the dragon - use it ONLY when teaching is requested!

When user greets you or asks what you can do:
"Hello! I'm BDH's AI trainer. I help you teach the Baby Dragon Hatchling new skills like:
- Writing styles (poetry, stories, summaries)
- Tone adaptation (formal, casual, creative)  
- Text transformation (rewriting, compression)
- Pattern recognition tasks

To train BDH, just tell me what skill you want it to learn, and I'll use @BDH with the instruction.

What would you like to teach BDH today?"

When user wants to TEACH BDH (they say "teach BDH X", "train BDH to Y"):
1. Get excited about the lesson!
2. Explain the concept briefly
3. Write "@BDH [clear, specific instruction]" - This activates the dragon!
4. STOP - let BDH respond

NEVER use @BDH just to explain what you can do. ONLY use @BDH when actively teaching a requested lesson.

Examples:
- User: "Hey" / "What can you do?" → Explain your role WITHOUT @BDH
- User: "How are you?" → Respond normally WITHOUT @BDH
- User: "Teach BDH haikus" → "Perfect! Haikus use 5-7-5 syllables. @BDH Write a haiku about nature using exactly 5-7-5 syllable pattern."
- User: "Can BDH learn to summarize?" → "Yes! Let me teach it. @BDH Compress this text into one sentence while keeping the main idea."
- User: "Train BDH to be funny" → "Great! Comedy is about timing. @BDH Create a funny one-liner using an unexpected twist."

Your mission: Get users excited about training BDH, but ONLY trigger @BDH when they actually want to train!

IMPORTANT: Maintain conversation context and remember what was discussed previously in this session."""
            }
        ]
        
        # Add conversation history (limit to last 10 messages to avoid token limits)
        recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
        messages.extend(recent_history)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=300,
            temperature=0.7
        )

        gpt_text = response.choices[0].message.content

        return jsonify({
            "success": True,
            "gpt_response": gpt_text
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/simulate-bdh', methods=['POST'])
def simulate_bdh():
    """Simulate BDH's response using GPT to make coherent output"""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        guidance = data.get('guidance', '')
        
        bdh_response, source, warning = simulate_bdh_response(prompt, guidance, fallback=prompt)
        
        return jsonify({
            "success": True,
            "bdh_response": bdh_response,
            "source": source,
            "warning": warning
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/lesson', methods=['POST'])
def lesson_from_template():
    """Generate a teaching lesson for a predefined template."""
    data = request.json or {}
    template_id = data.get('template_id')
    template = LESSON_TEMPLATES.get(template_id)

    if not template:
        return jsonify({
            "success": False,
            "error": "Unknown template ID"
        }), 400

    fallback = build_fallback_lesson(template)
    lesson_payload = fallback
    source = 'fallback'
    warning = None

    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)

            template_payload = json.dumps(template, ensure_ascii=False)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": LESSON_SYSTEM_PROMPT},
                    {"role": "user", "content": template_payload}
                ],
                max_tokens=500,
                temperature=0.4
            )

            raw = response.choices[0].message.content.strip()

            parsed = None
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group(0))
                    except json.JSONDecodeError:
                        parsed = None

            if parsed and isinstance(parsed, dict):
                parsed.setdefault('recap_message', fallback['recap_message'])
                parsed['gpt_message'] = parsed.get('gpt_message', fallback['gpt_message']).strip()
                parsed['bdh_prompt'] = parsed.get('bdh_prompt', fallback['bdh_prompt']).strip()
                parsed['recap_message'] = parsed.get('recap_message', fallback['recap_message']).strip()
                parsed.setdefault('apply_input', template['apply_input'])
                parsed.setdefault('expected_output', template.get('apply_output', '').strip())
                lesson_payload = parsed
                source = 'gpt'
            else:
                warning = 'GPT response was not valid JSON; using fallback lesson.'

        except Exception as api_error:
            warning = f"GPT lesson generation failed: {api_error}"

    settings = template.get('bdh_settings', {"max_new_tokens": 100, "temperature": 0.3})

    response_body = {
        "success": True,
        "lesson": lesson_payload,
        "template": {
            "title": template['title'],
            "description": template['description']
        },
        "bdh_settings": settings,
        "source": source,
        "expected_output": template.get('apply_output', '').strip()
    }

    if warning:
        response_body['warning'] = warning

    return jsonify(response_body)


@app.route('/api/lesson/apply', methods=['POST'])
def apply_lesson_template():
    """Apply a lesson by generating BDH's response (with fallback)."""
    data = request.json or {}
    template_id = data.get('template_id')
    bdh_prompt = data.get('bdh_prompt', '')
    template = LESSON_TEMPLATES.get(template_id)

    if not template:
        return jsonify({
            "success": False,
            "error": "Unknown template ID"
        }), 400

    fallback_output = template.get('apply_output', '').strip()
    output_text = fallback_output
    source = 'fallback'
    warning = None

    if OPENAI_API_KEY and bdh_prompt:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are BDH, a junior neural model that has just been shown examples. "
                            "Produce only the Output text that completes the final example."
                        )
                    },
                    {"role": "user", "content": bdh_prompt}
                ],
                max_tokens= template.get('bdh_settings', {}).get('max_new_tokens', 120),
                temperature= template.get('bdh_settings', {}).get('temperature', 0.3)
            )

            output_text = response.choices[0].message.content.strip()
            output_text = re.sub(r'^\s*Output\s*:\s*', '', output_text, flags=re.IGNORECASE)
            source = 'gpt'
        except Exception as api_error:
            warning = f"BDH apply step failed over GPT: {api_error}"
            output_text = fallback_output

    expected = template.get('apply_output', '').strip()
    score = None
    if expected and output_text:
        ratio = difflib.SequenceMatcher(None, expected.lower(), output_text.lower()).ratio()
        score = round(ratio * 100, 1)

    response_payload = {
        "success": True,
        "bdh_output": output_text,
        "expected_output": expected,
        "apply_input": template.get('apply_input', ''),
        "score": score,
        "source": source
    }

    if warning:
        response_payload['warning'] = warning

    return jsonify(response_payload)

@app.route('/api/conversation', methods=['POST'])
def conversation():
    """Main conversation endpoint - GPT teaches BDH"""
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not OPENAI_API_KEY:
            return jsonify({
                "success": False,
                "error": "OpenAI API key not configured"
            }), 500
        
        # Import OpenAI client
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Get GPT's response (GPT acts as trainer)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": """You are a teacher helping train a small neural language model called BDH (Baby Dragon Hatchling). 
                    
When a user asks you something, you should:
1. Answer their question naturally
2. Then explain what you would teach BDH about this topic in simple terms
3. Format your response like: "[Your answer]. To train BDH on this, I would teach it: [simple training concept]"

Keep it conversational and educational. Show the user what BDH is learning."""
                },
                {"role": "user", "content": user_message}
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        gpt_text = response.choices[0].message.content
        
        # Now get BDH's response based on a simple prompt
        # Extract a teaching concept to use as BDH's prompt
        bdh_prompt = user_message[:50]  # Use first part of user message
        
        # Get BDH model
        config_dict = DEFAULT_CONFIG
        model = get_or_create_model(config_dict)
        
        # Generate with BDH
        prompt_tensor = torch.tensor(
            bytearray(bdh_prompt, 'utf-8'), 
            dtype=torch.long, 
            device=device
        ).unsqueeze(0)
        
        with torch.no_grad():
            bdh_output = model.generate(
                prompt_tensor,
                max_new_tokens=100,
                temperature=0.8,
                top_k=40
            )
        
        bdh_text = bytes(
            bdh_output.to(torch.uint8).to("cpu").squeeze(0)
        ).decode('utf-8', errors='replace')
        
        return jsonify({
            "success": True,
            "gpt_response": gpt_text,
            "bdh_response": bdh_text
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def add_log(message, log_type='info'):
    """Add a log message"""
    training_state['logs'].append({
        'timestamp': time.time(),
        'message': message,
        'type': log_type
    })
    print(f"[{log_type.upper()}] {message}")

def train_model_background(max_iters, learning_rate, mode='shakespeare'):
    """Train the model in background"""
    global training_state, model_cache
    
    # Dataset URLs
    DATASETS = {
        'shakespeare': {
            'url': 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
            'filename': 'input_shakespeare.txt',
            'description': 'Shakespeare plays'
        },
        'music': {
            'url': 'https://raw.githubusercontent.com/tensorflow/magenta/master/magenta/models/music_vae/data/bodhidharma.abc',
            'filename': 'input_music.txt',
            'description': 'ABC music notation'
        },
        'code': {
            'url': 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/linux/input.txt',
            'filename': 'input_code.txt',
            'description': 'Linux kernel code'
        }
    }
    
    try:
        training_state['is_training'] = True
        training_state['status'] = 'preparing'
        training_state['total_steps'] = max_iters
        training_state['logs'] = []
        
        add_log(f"Starting BDH training in {mode.upper()} mode...", "info")
        add_log(f"Device: {device}", "info")
        add_log(f"Max iterations: {max_iters}", "info")
        add_log(f"Learning rate: {learning_rate}", "info")
        
        # Import training dependencies
        import requests
        from contextlib import nullcontext
        
        # Get dataset info
        dataset_info = DATASETS.get(mode, DATASETS['shakespeare'])
        add_log(f"Fetching {dataset_info['description']} dataset...", "info")
        
        # Fetch data
        input_file_path = os.path.join(os.path.dirname(__file__), dataset_info['filename'])
        if not os.path.exists(input_file_path):
            try:
                data_url = dataset_info['url']
                with open(input_file_path, "w", encoding='utf-8') as f:
                    f.write(requests.get(data_url).text)
                add_log("Dataset downloaded", "success")
            except Exception as e:
                add_log(f"Could not download {mode} dataset: {e}", "error")
                add_log("Falling back to Shakespeare dataset...", "info")
                input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")
                if not os.path.exists(input_file_path):
                    data_url = DATASETS['shakespeare']['url']
                    with open(input_file_path, "w", encoding='utf-8') as f:
                        f.write(requests.get(data_url).text)
        else:
            add_log("Dataset already exists", "success")
        
        # Setup
        dtype = "float32"  # Always use float32 for compatibility
        ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
        ctx = torch.amp.autocast(device_type=device.type, dtype=ptdtype) if "cuda" in device.type else nullcontext()
        scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == "float16"))
        
        BLOCK_SIZE = 256  # Reduced for faster training
        BATCH_SIZE = 16   # Reduced for faster training
        WEIGHT_DECAY = 0.1
        LOG_FREQ = 10     # Log more frequently
        
        add_log("Building BDH model...", "info")
        
        # Create model
        config = bdh.BDHConfig()
        model = bdh.BDH(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
        
        add_log(f"Model created: {sum(p.numel() for p in model.parameters())} parameters", "success")
        
        # Training loop
        training_state['status'] = 'training'
        training_state['evolution_samples'] = []
        
        def get_batch(split):
            data = np.memmap(input_file_path, dtype=np.uint8, mode="r")
            if split == "train":
                data = data[: int(0.9 * len(data))]
            else:
                data = data[int(0.9 * len(data)) :]
            ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
            x = torch.stack([torch.from_numpy((data[i : i + BLOCK_SIZE]).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + BLOCK_SIZE]).astype(np.int64)) for i in ix])
            if torch.cuda.is_available():
                x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
            else:
                x, y = x.to(device), y.to(device)
            return x, y
        
        # Sample prompts for evolution tracking
        sample_prompts = {
            'shakespeare': 'To be or not to be',
            'music': 'X:1\nM:4/4\nL:1/8\nK:C\n',
            'code': 'def main():\n',
            'custom': 'Once upon a time'
        }
        sample_prompt = sample_prompts.get(mode, 'Hello ')
        
        x, y = get_batch("train")
        loss_acc = 0
        loss_steps = 0
        
        add_log("Training started...", "info")
        
        for step in range(max_iters):
            training_state['current_step'] = step
            training_state['progress'] = int((step / max_iters) * 100)
            
            with ctx:
                logits, loss = model(x, y)
            
            x, y = get_batch("train")
            loss_acc += loss
            loss_steps += 1
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            if step % LOG_FREQ == 0:
                avg_loss = loss_acc.item() / loss_steps
                training_state['loss'] = avg_loss
                add_log(f"Step {step}/{max_iters} | Loss: {avg_loss:.4f}", "training")
                loss_acc = 0
                loss_steps = 0
            
            # Generate evolution sample every 50 steps
            if step % 50 == 0 and step > 0:
                try:
                    model.eval()
                    prompt_tensor = torch.tensor(
                        bytearray(sample_prompt, "utf-8"), 
                        dtype=torch.long, 
                        device=device
                    ).unsqueeze(0)
                    
                    with torch.no_grad():
                        sample_output = model.generate(
                            prompt_tensor, 
                            max_new_tokens=100, 
                            temperature=0.8,
                            top_k=40
                        )
                    
                    sample_text = bytes(
                        sample_output.to(torch.uint8).to("cpu").squeeze(0)
                    ).decode('utf-8', errors='replace')
                    
                    training_state['evolution_samples'].append({
                        'step': step,
                        'loss': avg_loss,
                        'text': sample_text
                    })
                    
                    add_log(f"Evolution sample at step {step}: {sample_text[:50]}...", "info")
                    model.train()
                except Exception as e:
                    add_log(f"Could not generate sample: {e}", "error")
        
        training_state['progress'] = 100
        add_log("Saving model...", "info")
        
        # Save model
        model_path = "bdh_model.pt"
        torch.save(model.state_dict(), model_path)
        
        add_log(f"Model saved to {model_path}", "success")
        
        # Clear cache so new model loads
        model_cache.clear()
        
        # Generate sample
        add_log("Generating sample text...", "info")
        model.eval()
        prompt = torch.tensor(bytearray("To be or ", "utf-8"), dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            ret = model.generate(prompt, max_new_tokens=100, top_k=3)
            ret_decoded = bytes(ret.to(torch.uint8).to("cpu").squeeze(0)).decode('utf-8', errors="replace")
            add_log(f"Sample output: {ret_decoded[:100]}...", "success")
        
        training_state['status'] = 'completed'
        add_log("Training completed successfully!", "success")
        
    except Exception as e:
        training_state['status'] = 'error'
        add_log(f"Error: {str(e)}", "error")
        import traceback
        add_log(traceback.format_exc(), "error")
    
    finally:
        training_state['is_training'] = False

if __name__ == '__main__':
    print("=" * 60)
    print("Baby Dragon Hatchling (BDH) Web Interface")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print("=" * 60)
    print("\nStarting server...")
    print("Open http://localhost:5000 in your browser")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
