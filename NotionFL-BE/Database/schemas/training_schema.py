from mongoengine import Document, IntField, BinaryField,  StringField, DateTimeField, DictField, ListField, EmbeddedDocument, EmbeddedDocumentField, FileField

class ClientEvaluation(EmbeddedDocument):
    evaluation_text = StringField()
    shap_plot = BinaryField()
    
class EvaluationPlot(EmbeddedDocument):
    round = StringField()
    plot = FileField()

class ClientModel(EmbeddedDocument):
    round_num = StringField(required=True)
    model_state = BinaryField(required=True)
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
    model_state = BinaryField()

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
    plots = ListField(FileField())

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
    client_training_logs = DictField()  # Stores training logs by client_id
    client_evaluation_logs = DictField()  # Stores evaluation logs by client_id and round
    client_models = ListField(EmbeddedDocumentField(ClientModel))
    privacy_explanations = ListField(EmbeddedDocumentField(PrivacyExplanation))
    differential_privacy_logs = ListField(DictField())
    aggregation_plots = ListField(FileField())
    secure_aggregation_logs = DictField()
    global_model_metrics = ListField(DictField())
    contribution_metrics = EmbeddedDocumentField(ContributionMetric)
    incentives_data = EmbeddedDocumentField(IncentivesData)
    incentive_explanation = EmbeddedDocumentField(IncentiveExplanation)
    global_evaluation = EmbeddedDocumentField(GlobalModelEvaluation)
    
    
    def to_json(self):
        # Update to_json method to include new fields
        return {
            "training_id": self.training_id,
            "initiator": self.initiator,
            "config": self.config,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "logs": self.logs,
            "aggregation": self.aggregation,
            "clients": [client.to_mongo() for client in self.clients],  # Convert list of EmbeddedDocuments to list of dictionaries
            "global_data": self.global_data.to_mongo() if self.global_data else None,
            "client_training_logs": self.client_training_logs,
            "client_evaluation_logs": self.client_evaluation_logs,
            "global_evaluation": {
                "evaluation_text": self.global_evaluation.evaluation_text,
                "shap_plot": self.global_evaluation.shap_plot.grid_id if self.global_evaluation.shap_plot else None,
                "cm_plot": self.global_evaluation.cm_plot.grid_id if self.global_evaluation.cm_plot else None,
            } if self.global_evaluation else None,
            "clients": [
                {
                    "client_id": client.client_id,
                    "contributions": client.contributions,
                    "evaluations": [eval.to_mongo() for eval in client.evaluations],
                    "model_comparisons": [
                        {
                            "explanation": comparison.explanation,
                            "comparison_plot": comparison.comparison_plot.grid_id if comparison.comparison_plot else None
                        }
                        for comparison in client.model_comparisons
                    ],
                    "evaluation_plots": [
                        {
                            "round": plot.round,
                            "plot": plot.plot.grid_id if plot.plot else None
                        }
                        for plot in client.evaluation_plots
                    ]
                }
                for client in self.clients
            ] if self.clients else None,
        }
