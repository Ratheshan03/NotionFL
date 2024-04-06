from mongoengine import Document, StringField, DateTimeField, DictField, FloatField, IntField

class TrainingModel(Document):
    training_id = StringField(required=True, unique=True)
    initiator = DictField(required=True)  
    config = DictField(required=True)     
    status = StringField(required=True, default='Pending')
    start_time = DateTimeField()
    end_time = DateTimeField()
    
    def to_json(self):
        return {
            "training_id": self.training_id,
            "initiator": self.initiator,
            "config": self.config,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None
        }
    
