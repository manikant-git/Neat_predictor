import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import json
from typing import Dict, List, Optional
import logging

class NEETRankPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
        
    def process_quiz_data(self, current_quiz: Dict, historical_quizzes: List[Dict]) -> pd.DataFrame:
        """
        Process current and historical quiz data into features for analysis
        """
        # Process current quiz
        current_features = {
            'current_score': current_quiz['score'],
            'topics_attempted': len(set(q['topic'] for q in current_quiz['questions'])),
            'accuracy': sum(1 for q in current_quiz['questions'] if q['correct']) / len(current_quiz['questions'])
        }
        
        # Process topic-wise performance
        topic_performance = {}
        for question in current_quiz['questions']:
            topic = question['topic']
            if topic not in topic_performance:
                topic_performance[topic] = {'correct': 0, 'total': 0}
            topic_performance[topic]['total'] += 1
            if question['correct']:
                topic_performance[topic]['correct'] += 1
        
        # Calculate topic-wise accuracy
        for topic, perf in topic_performance.items():
            current_features[f'topic_{topic}_accuracy'] = perf['correct'] / perf['total']
            
        # Process historical quizzes
        if historical_quizzes:
            hist_scores = [quiz['score'] for quiz in historical_quizzes]
            current_features.update({
                'avg_historical_score': np.mean(hist_scores),
                'score_trend': np.polyfit(range(len(hist_scores)), hist_scores, 1)[0],
                'score_volatility': np.std(hist_scores)
            })
            
        return pd.DataFrame([current_features])
    
    def train_model(self, training_data: pd.DataFrame, actual_ranks: List[int]):
        """
        Train the rank prediction model using historical data
        """
        try:
            X = self.scaler.fit_transform(training_data)
            self.model.fit(X, actual_ranks)
            self.logger.info("Model training completed successfully")
        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            raise
            
    def predict_rank(self, features: pd.DataFrame) -> int:
        """
        Predict NEET rank based on quiz performance
        """
        try:
            X = self.scaler.transform(features)
            predicted_rank = self.model.predict(X)[0]
            return int(predicted_rank)
        except Exception as e:
            self.logger.error(f"Error in rank prediction: {str(e)}")
            raise
            
    def analyze_performance(self, current_quiz: Dict, historical_quizzes: List[Dict]) -> Dict:
        """
        Generate detailed performance analysis and insights
        """
        analysis = {
            'weak_areas': [],
            'improvement_trends': {},
            'performance_summary': {}
        }
        
        # Identify weak areas
        topic_performance = {}
        for question in current_quiz['questions']:
            topic = question['topic']
            if topic not in topic_performance:
                topic_performance[topic] = {'correct': 0, 'total': 0}
            topic_performance[topic]['total'] += 1
            if question['correct']:
                topic_performance[topic]['correct'] += 1
                
        for topic, perf in topic_performance.items():
            accuracy = perf['correct'] / perf['total']
            if accuracy < 0.6:  # Threshold for weak areas
                analysis['weak_areas'].append({
                    'topic': topic,
                    'accuracy': accuracy,
                    'total_questions': perf['total']
                })
                
        # Analyze improvement trends
        if historical_quizzes:
            topic_trends = {}
            for quiz in historical_quizzes:
                for question in quiz['questions']:
                    topic = question['topic']
                    if topic not in topic_trends:
                        topic_trends[topic] = []
                    topic_trends[topic].append(1 if question['correct'] else 0)
                    
            for topic, scores in topic_trends.items():
                trend = np.polyfit(range(len(scores)), scores, 1)[0]
                analysis['improvement_trends'][topic] = {
                    'trend': trend,
                    'interpretation': 'improving' if trend > 0 else 'declining'
                }
                
        return analysis
    
    def predict_college(self, predicted_rank: int, college_data: List[Dict]) -> List[Dict]:
        """
        Predict potential colleges based on predicted rank
        """
        eligible_colleges = []
        for college in college_data:
            if predicted_rank <= college['last_rank']:
                eligible_colleges.append({
                    'name': college['name'],
                    'probability': self._calculate_admission_probability(
                        predicted_rank, 
                        college['last_rank'],
                        college['trend']
                    )
                })
        
        return sorted(eligible_colleges, key=lambda x: x['probability'], reverse=True)
    
    def _calculate_admission_probability(self, predicted_rank: int, cutoff_rank: int, trend: float) -> float:
        """
        Calculate probability of admission based on rank difference and historical trends
        """
        rank_difference = cutoff_rank - predicted_rank
        base_probability = min(1.0, max(0.0, rank_difference / cutoff_rank + 0.5))
        trend_factor = 1 + (0.1 * trend)  # Adjust probability based on historical trend
        return min(1.0, base_probability * trend_factor)

def create_performance_report(predictor: NEETRankPredictor, student_data: Dict) -> Dict:
    """
    Generate a comprehensive performance report
    """
    current_quiz = student_data['current_quiz']
    historical_quizzes = student_data['historical_quizzes']
    
    # Process data and generate predictions
    features = predictor.process_quiz_data(current_quiz, historical_quizzes)
    predicted_rank = predictor.predict_rank(features)
    analysis = predictor.analyze_performance(current_quiz, historical_quizzes)
    
    # Generate report
    report = {
        'predicted_rank': predicted_rank,
        'performance_analysis': analysis,
        'recommendations': generate_recommendations(analysis),
        'improvement_plan': create_improvement_plan(analysis['weak_areas'])
    }
    
    return report

def generate_recommendations(analysis: Dict) -> List[str]:
    """
    Generate personalized recommendations based on performance analysis
    """
    recommendations = []
    
    # Add recommendations based on weak areas
    for area in analysis['weak_areas']:
        recommendations.append(f"Focus on strengthening {area['topic']} - current accuracy: {area['accuracy']:.1%}")
        
    # Add recommendations based on improvement trends
    for topic, trend in analysis['improvement_trends'].items():
        if trend['interpretation'] == 'declining':
            recommendations.append(f"Increase practice frequency for {topic} to reverse declining trend")
            
    return recommendations

def create_improvement_plan(weak_areas: List[Dict]) -> Dict:
    """
    Create a structured improvement plan based on identified weak areas
    """
    plan = {
        'short_term': [],
        'medium_term': [],
        'long_term': []
    }
    
    # Prioritize weak areas by severity
    sorted_areas = sorted(weak_areas, key=lambda x: x['accuracy'])
    
    for i, area in enumerate(sorted_areas):
        if i < len(sorted_areas) // 3:
            plan['short_term'].append({
                'topic': area['topic'],
                'target': 'Improve accuracy by 20% in 2 weeks',
                'actions': [
                    'Daily practice questions',
                    'Review fundamental concepts',
                    'Take topic-specific mini-tests'
                ]
            })
        elif i < 2 * len(sorted_areas) // 3:
            plan['medium_term'].append({
                'topic': area['topic'],
                'target': 'Achieve 70% accuracy in 1 month',
                'actions': [
                    'Weekly topic reviews',
                    'Practice with varied difficulty levels',
                    'Error analysis and concept mapping'
                ]
            })
        else:
            plan['long_term'].append({
                'topic': area['topic'],
                'target': 'Master the topic in 2 months',
                'actions': [
                    'Deep concept study',
                    'Advanced problem solving',
                    'Regular revision and assessment'
                ]
            })
            
    return plan
