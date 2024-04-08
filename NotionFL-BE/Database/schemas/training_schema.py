import bson
from mongoengine import Document, ObjectIdField, IntField, ReferenceField, BinaryField,  StringField, DateTimeField, DictField, ListField, EmbeddedDocument, EmbeddedDocumentField, FileField

class ClientEvaluation(EmbeddedDocument):
    evaluation_text = StringField()
    shap_plot = BinaryField()
    
class EvaluationPlot(EmbeddedDocument):
    round = StringField()
    plot = FileField()

class ClientModel(EmbeddedDocument):
    round_num = StringField(required=True)
    model_state = StringField(required=True)
    suffix = StringField()
    
class ModelComparison(EmbeddedDocument):
    explanation = StringField()
    comparison_plot = FileField() 

class ClientData(EmbeddedDocument):
    client_id = StringField(required=True)
    contributions = ListField(StringField())
    evaluations = ListField(EmbeddedDocumentField(ClientEvaluation))
    models = ListField(EmbeddedDocumentField(ClientModel))  # Existing field
    updates = ListField(BinaryField())
    model_comparisons = ListField(EmbeddedDocumentField(ModelComparison))
    evaluation_plots = ListField(EmbeddedDocumentField(EvaluationPlot))
    
class GlobalModel(EmbeddedDocument):
    round = StringField(required=True)
    model_state = StringField() 

class GlobalModelData(EmbeddedDocument):
    evaluations = StringField()
    models = ListField(EmbeddedDocumentField(GlobalModel))
    privacy_explanations = ListField(StringField())
    
class GlobalModelEvaluation(EmbeddedDocument):
    evaluation_text = StringField()
    shap_plot = FileField()
    cm_plot = FileField()
    
class PrivacyExplanation(EmbeddedDocument):
    text = StringField()
    visualization = BinaryField()
    
class ContributionMetric(EmbeddedDocument):
    shapley_values = DictField()
    shapley_plot = FileField()
    
class IncentivesData(EmbeddedDocument):
    incentives_json = DictField()
    incentive_plot = FileField()
    
class IncentiveExplanation(EmbeddedDocument):
    explanation_text = StringField()
    plots = ListField(ObjectIdField())
    
class AggregationPlot(Document):
    plot = FileField()

class TrainingModel(Document):
    training_id = StringField(required=True, unique=True)
    initiator = DictField(required=True)  
    config = DictField(required=True)
    status = StringField(required=True, default='Pending')
    start_time = DateTimeField()
    end_time = DateTimeField()
    logs = StringField()
    aggregation = ListField(DictField())
    clients = ListField(EmbeddedDocumentField(ClientData))
    global_data = EmbeddedDocumentField(GlobalModelData)
    client_training_logs = DictField()  
    client_evaluation_logs = DictField()
    client_models = ListField(EmbeddedDocumentField(ClientModel))
    privacy_explanations = ListField(EmbeddedDocumentField(PrivacyExplanation))
    differential_privacy_logs = ListField(DictField())
    aggregation_plots = ListField(ReferenceField(AggregationPlot))
    secure_aggregation_logs = DictField()
    global_model_metrics = ListField(DictField())
    contribution_metrics = EmbeddedDocumentField(ContributionMetric)
    incentives_data = EmbeddedDocumentField(IncentivesData)
    global_evaluation = EmbeddedDocumentField(GlobalModelEvaluation)
    incentive_explanation = EmbeddedDocumentField(IncentiveExplanation)
    
    
    def to_json(self):
        # Updated to_json method to include new fields
        return {
            "training_id": self.training_id,
            "initiator": self.initiator,
            "config": self.config,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "logs": self.logs,
            "aggregation": self.aggregation,
            "clients": [
                {
                    "client_id": client.client_id,
                    "contributions": client.contributions,
                    "evaluations": [eval.to_mongo() for eval in client.evaluations],
                    "models": [model.to_mongo() for model in client.models],
                    "updates": [bson.binary.Binary(update) for update in client.updates],
                    "model_comparisons": [comparison.to_mongo() for comparison in client.model_comparisons],
                    "evaluation_plots": [plot.to_mongo() for plot in client.evaluation_plots],
                }
                for client in self.clients
            ] if self.clients else None,
            "global_data": self.global_data.to_mongo() if self.global_data else None,
            "client_training_logs": self.client_training_logs,
            "client_evaluation_logs": self.client_evaluation_logs,
            "differential_privacy_logs": self.differential_privacy_logs,
            "aggregation_plots": [plot.to_mongo() for plot in self.aggregation_plots],
            "secure_aggregation_logs": self.secure_aggregation_logs,
            "global_model_metrics": self.global_model_metrics,
            "contribution_metrics": self.contribution_metrics.to_mongo() if self.contribution_metrics else None,
            "incentives_data": self.incentives_data.to_mongo() if self.incentives_data else None,
            "global_evaluation": self.global_evaluation.to_mongo() if self.global_evaluation else None,
            "incentive_explanation": self.incentive_explanation.to_mongo() if self.incentive_explanation else None,
            "privacy_explanations": [explanation.to_mongo() for explanation in self.privacy_explanations],
        }
