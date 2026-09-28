from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return """
    <html>
        <head>
            <title>SoundGuard</title>
        </head>
        <body>
            <h1>SoundGuard</h1>
            <h2>On-Device Fan Health Monitor</h2>
            <p>AI-powered abnormal fan sound detection for Snapdragon laptops.</p>
            <p>SoundGuard is running successfully.</p>
        </body>
    </html>
    """