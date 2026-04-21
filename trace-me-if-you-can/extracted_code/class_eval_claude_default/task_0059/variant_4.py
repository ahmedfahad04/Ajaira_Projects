from datetime import datetime
import numpy as np

class MovieBookingSystem:
    def __init__(self):
        self.movies = []
        self._movie_lookup = {}

    def add_movie(self, name, price, start_time, end_time, n):
        movie = {
            'name': name,
            'price': price,
            'start_time': datetime.strptime(start_time, '%H:%M'),
            'end_time': datetime.strptime(end_time, '%H:%M'),
            'seats': np.zeros((n, n))
        }
        self.movies.append(movie)
        self._movie_lookup[name] = len(self.movies) - 1

    def book_ticket(self, name, seats_to_book):
        try:
            movie_idx = self._movie_lookup[name]
            movie = self.movies[movie_idx]
        except KeyError:
            return "Movie not found."
        
        # Pre-validate booking using generator expression
        if any(movie['seats'][row, col] != 0 for row, col in seats_to_book):
            return "Booking failed."
        
        # Batch update using numpy advanced indexing
        rows, cols = zip(*seats_to_book) if seats_to_book else ([], [])
        movie['seats'][rows, cols] = 1
        return "Booking success."

    def available_movies(self, start_time, end_time):
        start_dt = datetime.strptime(start_time, '%H:%M')
        end_dt = datetime.strptime(end_time, '%H:%M')
        
        def is_within_timeframe(movie):
            return start_dt <= movie['start_time'] and movie['end_time'] <= end_dt
        
        return [movie['name'] for movie in self.movies if is_within_timeframe(movie)]
