import pandas as pd
import numpy as np
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class ExpenseClassifier:
    """ML Expense Classifier without NLTK"""
    
    def __init__(self):
        # Custom stopwords list
        self.stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
            'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
            'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
            'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
            'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
            'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
            'between', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
            'off', 'over', 'under', 'again', 'further', 'then', 'once',
            'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
            'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
        }
        
        # TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 3),  # Capture "going to" as bigram
            stop_words=list(self.stop_words),
            min_df=2,
            max_df=0.9
        )
        
        # Classifier
        self.classifier = None
        self.categories = ['food', 'transport', 'shopping', 'utilities', 'entertainment']
        
        # Category keywords for fallback
        self.category_keywords = {
            'food': ['restaurant', 'cafe', 'food', 'pizza', 'burger', 'coffee', 
                    'tea', 'dinner', 'lunch', 'breakfast', 'meal', 'eat', 'drink',
                    'groceries', 'vegetables', 'fruits', 'snack'],
            'transport': ['uber', 'taxi', 'bus', 'train', 'flight', 'travel', 
                         'car', 'fuel', 'petrol', 'diesel', 'auto', 'rickshaw',
                         'metro', 'airport', 'station', 'fare', 'ticket'],
            'shopping': ['buy', 'purchase', 'shopping', 'mall', 'store', 'shop',
                        'clothes', 'shoes', 'dress', 'shirt', 'pant', 'jeans',
                        'electronics', 'phone', 'mobile', 'laptop', 'watch'],
            'utilities': ['bill', 'electricity', 'water', 'internet', 'gas',
                         'recharge', 'wifi', 'broadband', 'cable', 'tv',
                         'mobile', 'phone', 'payment', 'subscription'],
            'entertainment': ['movie', 'netflix', 'concert', 'game', 'show',
                             'ticket', 'theater', 'cinema', 'music', 'sports',
                             'event', 'party', 'celebration', 'fun']
        }
    
    def preprocess_text(self, text):
        """Simple text preprocessing without NLTK"""
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove special characters (keep spaces)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def generate_training_data(self):
        """Generate comprehensive training data"""
        data = []
        
        # ===== FOOD EXAMPLES =====
        food_data = [
            ('pizza at dominos', 'food'),
            ('burger from mcdonalds', 'food'),
            ('coffee at starbucks', 'food'),
            ('lunch at restaurant', 'food'),
            ('dinner with friends', 'food'),
            ('breakfast at hotel', 'food'),
            ('groceries from supermarket', 'food'),
            ('vegetables from market', 'food'),
            ('food delivery swiggy', 'food'),
            ('takeaway dinner', 'food'),
            ('ice cream parlour', 'food'),
            ('chocolate purchase', 'food'),
            ('tea at local cafe', 'food'),
            ('juice from juice center', 'food'),
            ('bakery items bread', 'food'),
            ('food court spending', 'food'),
            ('restaurant bill payment', 'food'),
            ('dining at five star', 'food'),
            ('fast food burger king', 'food'),
            ('pizza hut dinner', 'food'),
            ('kfc chicken bucket', 'food'),
            ('subway sandwich meal', 'food'),
            ('mcdonalds combo meal', 'food'),
            ('donuts from dunkin', 'food'),
            ('pastry from bakery shop', 'food'),
            ('coffee beans purchase', 'food'),
            ('organic food store', 'food'),
            ('milk and eggs', 'food'),
            ('kitchen supplies', 'food'),
            ('cooking oil', 'food')
        ]
        
        # ===== TRANSPORT EXAMPLES =====
        transport_data = [
            ('going to ambala', 'transport'),
            ('uber ride to office', 'transport'),
            ('bus ticket to delhi', 'transport'),
            ('train journey mumbai', 'transport'),
            ('taxi to airport', 'transport'),
            ('fuel for car', 'transport'),
            ('petrol pump fill', 'transport'),
            ('metro card recharge', 'transport'),
            ('auto rickshaw fare', 'transport'),
            ('airport taxi service', 'transport'),
            ('going to market', 'transport'),
            ('travel to bangalore', 'transport'),
            ('ola cab ride', 'transport'),
            ('car wash service', 'transport'),
            ('bike petrol fill', 'transport'),
            ('flight tickets booking', 'transport'),
            ('train reservation charges', 'transport'),
            ('bus pass renewal', 'transport'),
            ('cab to railway station', 'transport'),
            ('travel insurance purchase', 'transport'),
            ('road trip expenses', 'transport'),
            ('commute to work daily', 'transport'),
            ('school bus fee', 'transport'),
            ('office cab service', 'transport'),
            ('truck diesel fill', 'transport'),
            ('bicycle repair service', 'transport'),
            ('scooter service center', 'transport'),
            ('car loan emi', 'transport'),
            ('vehicle insurance premium', 'transport'),
            ('parking charges mall', 'transport')
        ]
        
        # ===== SHOPPING EXAMPLES =====
        shopping_data = [
            ('buy new clothes', 'shopping'),
            ('shopping at big bazaar', 'shopping'),
            ('purchase mobile phone', 'shopping'),
            ('online shopping amazon', 'shopping'),
            ('electronics from croma', 'shopping'),
            ('grocery shopping list', 'shopping'),
            ('book purchase crossword', 'shopping'),
            ('watch from titan showroom', 'shopping'),
            ('jewelry from tanishq', 'shopping'),
            ('cosmetics from nykaa', 'shopping'),
            ('furniture from ikea', 'shopping'),
            ('stationery items purchase', 'shopping'),
            ('kitchen utensils buy', 'shopping'),
            ('home decor shopping', 'shopping'),
            ('sports equipment purchase', 'shopping'),
            ('toys for kids', 'shopping'),
            ('gift shopping birthday', 'shopping'),
            ('apparel from myntra', 'shopping'),
            ('footwear from bata', 'shopping'),
            ('bags from luggage shop', 'shopping'),
            ('perfume purchase mall', 'shopping'),
            ('sunglasses from rayban', 'shopping'),
            ('laptop from dell store', 'shopping'),
            ('mobile accessories case', 'shopping'),
            ('computer parts ram', 'shopping'),
            ('camera equipment dslr', 'shopping'),
            ('musical instruments guitar', 'shopping'),
            ('art supplies canvas', 'shopping'),
            ('craft materials diy', 'shopping'),
            ('garden tools purchase', 'shopping')
        ]
        
        # ===== UTILITIES EXAMPLES =====
        utilities_data = [
            ('electricity bill payment', 'utilities'),
            ('water bill online', 'utilities'),
            ('internet broadband recharge', 'utilities'),
            ('gas cylinder delivery', 'utilities'),
            ('mobile phone recharge', 'utilities'),
            ('dth subscription renewal', 'utilities'),
            ('landline bill payment', 'utilities'),
            ('wifi router purchase', 'utilities'),
            ('cable tv bill', 'utilities'),
            ('newspaper subscription', 'utilities'),
            ('magazine renewal', 'utilities'),
            ('software subscription', 'utilities'),
            ('cloud storage payment', 'utilities'),
            ('domain name renewal', 'utilities'),
            ('hosting charges server', 'utilities'),
            ('security system payment', 'utilities'),
            ('maintenance charges society', 'utilities'),
            ('house tax payment', 'utilities'),
            ('property tax online', 'utilities'),
            ('insurance premium life', 'utilities'),
            ('loan emi payment bank', 'utilities'),
            ('credit card bill', 'utilities'),
            ('bank charges atm', 'utilities'),
            ('atm withdrawal charges', 'utilities'),
            ('cheque book charges', 'utilities'),
            ('bank locker rent', 'utilities'),
            ('mutual fund sip', 'utilities'),
            ('stock trading charges', 'utilities'),
            ('investment platform fees', 'utilities'),
            ('tax filing charges', 'utilities')
        ]
        
        # ===== ENTERTAINMENT EXAMPLES =====
        entertainment_data = [
            ('movie tickets pvr', 'entertainment'),
            ('netflix monthly subscription', 'entertainment'),
            ('amazon prime video', 'entertainment'),
            ('hotstar premium subscription', 'entertainment'),
            ('concert tickets booking', 'entertainment'),
            ('theater play tickets', 'entertainment'),
            ('circus show tickets', 'entertainment'),
            ('museum entry fees', 'entertainment'),
            ('amusement park visit', 'entertainment'),
            ('video game purchase', 'entertainment'),
            ('ps5 game cd', 'entertainment'),
            ('xbox subscription', 'entertainment'),
            ('spotify premium', 'entertainment'),
            ('youtube premium', 'entertainment'),
            ('gym membership', 'entertainment'),
            ('swimming pool charges', 'entertainment'),
            ('sports club fees', 'entertainment'),
            ('dance class fees', 'entertainment'),
            ('music concert live', 'entertainment'),
            ('standup comedy show', 'entertainment'),
            ('exhibition entry ticket', 'entertainment'),
            ('fair visit expenses', 'entertainment'),
            ('picnic expenses park', 'entertainment'),
            ('outing with friends', 'entertainment'),
            ('party expenses birthday', 'entertainment'),
            ('birthday celebration cake', 'entertainment'),
            ('anniversary dinner special', 'entertainment'),
            ('date night expenses', 'entertainment'),
            ('weekend getaway trip', 'entertainment'),
            ('holiday spending vacation', 'entertainment')
        ]
        
        # Combine all data
        data.extend(food_data)
        data.extend(transport_data)
        data.extend(shopping_data)
        data.extend(utilities_data)
        data.extend(entertainment_data)
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=['text', 'category'])
        df['cleaned_text'] = df['text'].apply(self.preprocess_text)
        
        return df
    
    def train_model(self):
        """Train the ML model"""
        print("📊 Generating training data...")
        df = self.generate_training_data()
        
        print(f"✅ Generated {len(df)} training examples")
        print(f"📈 Category distribution:")
        print(df['category'].value_counts())
        
        # Vectorize text
        X = self.vectorizer.fit_transform(df['cleaned_text'])
        y = df['category']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n🤖 Training ML models...")
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Testing samples: {X_test.shape[0]}")
        
        # Train multiple models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=150,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'SVM': SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=42,
                class_weight='balanced'
            ),
            'Naive Bayes': MultinomialNB(alpha=0.1)
        }
        
        best_model = None
        best_score = 0
        best_name = ""
        
        for name, model in models.items():
            print(f"   Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            print(f"   ✅ {name} Accuracy: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_model = model
                best_name = name
        
        self.classifier = best_model
        
        print(f"\n🎯 Best model: {best_name}")
        print(f"📊 Best accuracy: {best_score:.2%}")
        
        # Detailed evaluation
        print("\n📋 Classification Report:")
        y_pred = self.classifier.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=self.categories))
        
        # Save model
        self.save_model()
        
        return best_score
    
    def predict(self, text):
        """Predict category for input text"""
        # Rule-based for "going to" patterns (high confidence)
        text_lower = text.lower()
        
        if 'going to' in text_lower:
            destination = text_lower.split('going to')[-1].strip()
            if destination:
                destination = destination.title()
            
            return {
                'category': 'transport',
                'confidence': 0.98,
                'explanation': f"Detected travel pattern: 'going to {destination}'",
                'text': text,
                'model_used': 'rule_based',
                'destination': destination
            }
        
        # Try ML prediction
        if self.classifier is None:
            print("⚠️ ML model not loaded, using keyword fallback")
            return self.keyword_based_prediction(text)
        
        try:
            # Preprocess and vectorize
            cleaned = self.preprocess_text(text)
            
            if not cleaned.strip():
                return self.keyword_based_prediction(text)
            
            vector = self.vectorizer.transform([cleaned])
            
            # Get prediction
            category = self.classifier.predict(vector)[0]
            probabilities = self.classifier.predict_proba(vector)[0]
            confidence = probabilities.max()
            
            # Get explanation
            explanation = self.generate_explanation(text, category)
            
            return {
                'category': category,
                'confidence': float(confidence),
                'explanation': explanation,
                'text': text,
                'model_used': 'ml',
                'probabilities': {
                    cat: float(prob) 
                    for cat, prob in zip(self.classifier.classes_, probabilities)
                }
            }
            
        except Exception as e:
            print(f"ML prediction error: {e}")
            return self.keyword_based_prediction(text)
    
    def keyword_based_prediction(self, text):
        """Fallback to keyword matching"""
        text_lower = text.lower()
        
        scores = {}
        for category, keywords in self.category_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 0:
                scores[category] = matches
        
        if scores:
            best_category = max(scores, key=scores.get)
            confidence = min(scores[best_category] * 0.15, 0.85)
            
            explanation = f"Found {scores[best_category]} matching keywords: "
            matched_keywords = [kw for kw in self.category_keywords[best_category] 
                              if kw in text_lower]
            explanation += ', '.join(matched_keywords[:3])
            
            return {
                'category': best_category,
                'confidence': confidence,
                'explanation': explanation,
                'text': text,
                'model_used': 'keyword_fallback'
            }
        
        # Default to 'other'
        return {
            'category': 'other',
            'confidence': 0.5,
            'explanation': "Could not determine category from text",
            'text': text,
            'model_used': 'default'
        }
    
    def generate_explanation(self, text, category):
        """Generate explanation for ML prediction"""
        explanations = {
            'food': "AI detected food and dining related patterns",
            'transport': "AI identified transportation and travel indicators",
            'shopping': "AI recognized shopping and purchase patterns",
            'utilities': "AI found utility bill and service payment patterns",
            'entertainment': "AI detected entertainment and leisure activity indicators"
        }
        
        base = explanations.get(category, "AI analyzed the text patterns")
        
        # Add specific details based on text
        text_lower = text.lower()
        specifics = []
        
        if 'bill' in text_lower and category == 'utilities':
            specifics.append("bill payment detected")
        elif 'buy' in text_lower or 'purchase' in text_lower:
            specifics.append("purchase action identified")
        elif 'at' in text_lower:
            specifics.append("location reference found")
        
        if specifics:
            return f"{base} ({', '.join(specifics)})"
        
        return base
    
    def save_model(self):
        """Save trained model to file"""
        try:
            joblib.dump({
                'vectorizer': self.vectorizer,
                'classifier': self.classifier,
                'categories': self.categories
            }, 'expense_model.pkl')
            print("💾 Model saved as 'expense_model.pkl'")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
    
    def load_model(self):
        """Load trained model from file"""
        try:
            model_data = joblib.load('expense_model.pkl')
            self.vectorizer = model_data['vectorizer']
            self.classifier = model_data['classifier']
            self.categories = model_data.get('categories', self.categories)
            print("📂 Model loaded successfully")
            return True
        except FileNotFoundError:
            print("⚠️ No saved model found")
            return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

# Global instance
expense_classifier = ExpenseClassifier()