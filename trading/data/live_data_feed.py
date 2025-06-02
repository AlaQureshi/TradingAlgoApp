import websocket
import json
import logging
import requests
from datetime import datetime
from threading import Thread, Event
import time

class PolygonWebSocket:
    def __init__(self, api_key, symbols, on_data_callback):
        self.api_key = api_key
        self.symbols = symbols
        self.on_data_callback = on_data_callback
        self.logger = logging.getLogger(__name__)
        self.connected = Event()
        self.authenticated = Event()
        self.data_received = Event()
        self.should_run = True
        self.ws = None
        self.rest_base_url = "https://api.polygon.io/v2"
        self.last_data_time = 0

    def connect(self):
        """Connect to Polygon.io websocket"""
        self.logger.info("Initializing websocket connection...")
        self.should_run = True
        try:
            # Get initial data via REST
            success = self._get_initial_data()
            if success:
                self.data_received.set()

            # Setup websocket
            self.ws = websocket.WebSocketApp(
                "wss://socket.polygon.io/stocks",
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            
            ws_thread = Thread(target=self._run_websocket)
            ws_thread.daemon = True
            ws_thread.start()
            
        except Exception as e:
            self.logger.error(f"Connection error: {e}")

    def _run_websocket(self):
        """Run websocket with automatic reconnection"""
        while self.should_run:
            try:
                self.ws.run_forever()
                if self.should_run:
                    self.logger.info("Websocket disconnected, reconnecting...")
                    time.sleep(5)  # Wait before reconnecting
            except Exception as e:
                self.logger.error(f"Websocket error: {e}")
                time.sleep(5)

    def _get_initial_data(self):
        """Get initial data via REST API"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            success = False
            
            for symbol in self.symbols:
                url = f"{self.rest_base_url}/last/trade/{symbol}"
                response = requests.get(url, headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'OK':
                        processed_data = {
                            'symbol': symbol,
                            'price': float(data['results']['p']),
                            'size': int(data['results']['s']),
                            'timestamp': datetime.fromtimestamp(data['results']['t']/1000)
                        }
                        self.on_data_callback(processed_data)
                        success = True
                        self.last_data_time = time.time()
                
            return success
        except Exception as e:
            self.logger.error(f"REST API error: {e}")
            return False

    def is_connected(self):
        """Check if we're receiving data"""
        return (self.data_received.is_set() and 
                (time.time() - self.last_data_time) < 30)  # 30 sec timeout

    def disconnect(self):
        """Properly close the connection"""
        self.should_run = False
        if self.ws:
            self.ws.close()
        self.connected.clear()
        self.authenticated.clear()
        self.data_received.clear()

    def on_open(self, ws):
        """Called when websocket connection opens"""
        self.logger.info("Websocket connection opened")
        auth_data = {"action": "auth", "params": self.api_key}
        ws.send(json.dumps(auth_data))

    def on_message(self, ws, message):
        """Handle incoming websocket messages"""
        try:
            data = json.loads(message)
            
            if isinstance(data, list) and len(data) > 0:
                msg = data[0]
                
                if msg.get('ev') == 'status':
                    if msg.get('status') == 'auth_success':
                        self.logger.info("Authentication successful")
                        self.authenticated.set()
                        self._subscribe(ws)
                    return
                    
                if msg.get('ev') in ['AM', 'A']:  # Minute aggregates
                    self.last_data_time = time.time()
                    self.data_received.set()
                    processed_data = {
                        'symbol': msg['sym'],
                        'price': float(msg['c']),
                        'size': int(msg['v']),
                        'timestamp': datetime.fromtimestamp(msg['e']/1000)
                    }
                    self.on_data_callback(processed_data)
                    
        except Exception as e:
            self.logger.error(f"Message processing error: {e}")

    def _subscribe(self, ws):
        """Subscribe to data streams"""
        channels = []
        for sym in self.symbols:
            channels.extend([f"AM.{sym}", f"A.{sym}"])
        
        subscribe_data = {
            "action": "subscribe",
            "params": ",".join(channels)
        }
        ws.send(json.dumps(subscribe_data))
        self.logger.info(f"Subscribed to channels: {channels}")

    def on_error(self, ws, error):
        """Handle websocket errors"""
        self.logger.error(f"Websocket error: {error}")
        self.connected.clear()
        self.authenticated.clear()

    def on_close(self, ws, close_status_code, close_msg):
        """Handle websocket connection close"""
        self.logger.info(f"Websocket connection closed: {close_msg}")
        self.connected.clear()
        self.authenticated.clear()
