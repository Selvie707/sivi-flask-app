from flask_sqlalchemy import SQLAlchemy
from uuid import uuid4

db = SQLAlchemy()

def get_uuid():
    return uuid4().hex

class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.String(11), primary_key=True, unique=True, default=get_uuid)
    username = db.Column(db.String(150))
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.Text, nullable=False)
    is_paid = db.Column(db.Boolean, default=False, nullable=False)
    role = db.Column(db.String(50), default="user", nullable=False)