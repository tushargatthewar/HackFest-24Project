from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

@app.route('/login', methods=['POST'])
def login():
    if request.method == 'POST':
        village_name = request.form['village_name']
        latitude, longitude = get_coordinates(village_name)
        if latitude is not None and longitude is not None:
            # Store latitude and longitude in session or database
            return render_template('map.html', latitude=latitude, longitude=longitude)
        else:
            return render_template('login.html', error='Invalid village name')

def get_coordinates(village_name):
    # Replace YOUR_API_KEY with your actual Google Maps API key
    api_key = 'YOUR_API_KEY'
    url = f'https://maps.googleapis.com/maps/api/geocode/json?address={village_name}&key={api_key}'

    response = requests.get(url)
    data = response.json()

    if data['status'] == 'OK':
        location = data['results'][0]['geometry']['location']
        latitude = location['lat']
        longitude = location['lng']
        return latitude, longitude
    else:
        return None, None

if __name__ == '__main__':
    app.run(debug=True)
