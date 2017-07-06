#from flask_wtf import Form
from flask_wtf import FlaskForm
from wtforms.fields import StringField, SelectField, SubmitField, TextAreaField, TextField, PasswordField
from wtforms.widgets import TextArea, TextInput, ListWidget
from wtforms.validators import DataRequired, ValidationError, Email, Length
from wtforms.fields.html5 import EmailField

class PostForm(FlaskForm):
	title = StringField(u'title', validators=[DataRequired()])
	speechName = StringField(u'Name', [DataRequired(), Length(min=1, max=8)])
	body = StringField(u'Text', widget=TextArea())
	party = SelectField('Political Leaning', choices=[('Conservative', 'Conservative'), ('Green', 'Green'), ('Liberal', 'Liberal'), ('Libertarian', 'Libertarian')])
	strength = SelectField('Strength', choices=[('Moderate', 'Moderate'), ('Strong', 'Strong')])
	sensitivity = SelectField('By Sentence Sensitivity', choices = [(0.7, 'High Sensitivity: Fewer, but stronger results'), (0.5, 'Medium Sensitivity: Moderate results'), (0.3, 'Low Sensitivity: Many results with many possible false positives')])
	
	
class citySelect(FlaskForm):
    title = StringField(u'title', validators=[DataRequired()])
    cityState = StringField(u'Text', widget=TextInput(), validators=[DataRequired()])
	
	
class twitter(FlaskForm):
	title = StringField(u'title', validators=[DataRequired()])
	query1 = StringField(u'Text', widget=TextInput())
	query2 = StringField(u'Text', widget=TextInput())
	query3 = StringField(u'Text', widget=TextInput())

	
class ContactForm(FlaskForm):
	name = TextField("Name", [DataRequired("Please enter your name.")])
	email = EmailField("Email", [DataRequired("Please enter your email address"), Email()])
	subject = TextField("Subject", [DataRequired("Please enter a subject.")])
	message = TextAreaField("Message", [DataRequired("Please enter a message")])
	submit = SubmitField("Send")
	
	
class LoginForm(FlaskForm):
	email = TextField("email", [DataRequired("Email address is required.")])
	password = PasswordField('New Password', [DataRequired("Password is required")])
	submit = SubmitField("Login")
	
	
class SignUp(FlaskForm):
	first = TextField("First Name", [DataRequired("First Name Required.")])
	last = TextField("Last Name", [DataRequired("Last Name Required.")])
	email = EmailField("Email", [DataRequired("Please enter your email address"), Email()])
	password = PasswordField('New Password', [DataRequired("Password is required")])