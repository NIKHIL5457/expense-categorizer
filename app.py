from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_model import expense_classifier

app = Flask(__name__, static_folder='static')

# Initialize model
print("🚀 Loading Expense Categorizer...")
if not expense_classifier.load_model():
    print("⚠️ No pre-trained model found. Training new one...")
    try:
        expense_classifier.train_model()
        print("✅ Model trained successfully")
    except Exception as e:
        print(f"❌ Training failed: {e}")

@app.route('/')
def home():
    """Main page"""
    return render_template('index.html')

@app.route('/api/categorize', methods=['POST'])
def categorize():
    """Categorize expense text"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                'success': False,
                'error': 'Please enter expense description'
            }), 400
        
        # Get prediction
        result = expense_classifier.predict(text)
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'expense-categorizer',
        'ml_model_loaded': expense_classifier.classifier is not None,
        'version': '1.0.0'
    })

@app.route('/api/train', methods=['POST'])
def train():
    """Train model via API"""
    try:
        accuracy = expense_classifier.train_model()
        return jsonify({
            'success': True,
            'accuracy': accuracy,
            'message': 'Model trained successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get list of categories"""
    return jsonify({
        'success': True,
        'categories': expense_classifier.categories,
        'count': len(expense_classifier.categories)
    })

# Serve static files
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/script.js')
def serve_js():
    return send_from_directory('.', 'script.js')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)