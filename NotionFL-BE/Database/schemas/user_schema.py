from mongoengine import Document, StringField, EmailField, BooleanField

class User(Document):
    username = StringField(required=True, unique=True)
    email = EmailField(required=True, unique=True)
    hashed_password = StringField(required=True)
    role = StringField(required=True, choices=('client', 'admin'))
    organization = StringField() 
    contact = StringField()       
    is_active = BooleanField(default=True)

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
