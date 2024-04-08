from mongoengine import Document, StringField, ListField,EmbeddedDocumentField, EmailField, BooleanField, EmbeddedDocument

class TrainingSession(EmbeddedDocument):
    training_id = StringField(required=True)
    client_id = StringField(required=True)
class User(Document):
    username = StringField(required=True, unique=True)
    email = EmailField(required=True, unique=True)
    hashed_password = StringField(required=True)
    role = StringField(required=True, choices=('client', 'admin'))
    organization = StringField() 
    contact = StringField()       
    is_active = BooleanField(default=True)
    trainings = ListField(EmbeddedDocumentField(TrainingSession)) 

    meta = {
        'collection': 'users',
        'indexes': [
            'email',
            'username',
            # Additional indexes as required.
        ],
    }
    
    def to_json(self):
        # Convert the embedded documents to JSON as well
        trainings_json = [training.to_mongo() for training in self.trainings]
        return {
            "id": str(self.id),
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "organization": self.organization,
            "contact": self.contact,
            "is_active": self.is_active,
            "trainings": trainings_json
        }
