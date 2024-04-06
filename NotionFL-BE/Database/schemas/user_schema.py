import datetime
from mongoengine import Document, ListField, StringField, IntField, EmailField, BooleanField, EmbeddedDocumentField, EmbeddedDocument

class TrainingSession(EmbeddedDocument):
    training_id = StringField(required=True)
    client_number = IntField(required=True)
    
class User(Document):
    username = StringField(required=True, unique=True)
    email = EmailField(required=True, unique=True)
    hashed_password = StringField(required=True)
    role = StringField(required=True, choices=('client', 'admin'))
    organization = StringField() 
    contact = StringField()       
    is_active = BooleanField(default=True)
    training_sessions = ListField(EmbeddedDocumentField(TrainingSession)) 

    meta = {
        'collection': 'users',
        'indexes': [
            'email',
            'username',
            # Additional indexes as required.
        ],
    }
    
    def to_json(self):
        return {
            "id": str(self.id),
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "organization": self.organization,
            "contact": self.contact,
            "is_active": self.is_active
        }
