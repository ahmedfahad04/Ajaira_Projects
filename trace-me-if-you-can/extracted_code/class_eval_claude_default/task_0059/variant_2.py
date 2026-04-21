from datetime import datetime
import numpy as np

class Movie:
    def __init__(self, name, price, start_time, end_time, n):
        self.name = name
        self.price = price
        self.start_time = datetime.strptime(start_time, '%H:%M')
        self.end_time = datetime.strptime(end_time, '%H:%M')
        self.seats = np.zeros((n, n))
    
    def is_seat_available(self, row, col):
        return self.seats[row][col] == 0
    
    def book_seat(self, row, col):
        self.seats[row][col] = 1

class MovieBookingSystem:
    def __init__(self):
        self.movies = []

    def add_movie(self, name, price, start_time, end_time, n):
        movie = Movie(name, price, start_time, end_time, n)
        self.movies.append(movie)

    def _find_movie(self, name):
        return next((movie for movie in self.movies if movie.name == name), None)

    def book_ticket(self, name, seats_to_book):
        movie = self._find_movie(name)
        if not movie:
            return "Movie not found."
        
        if not all(movie.is_seat_available(row, col) for row, col in seats_to_book):
            return "Booking failed."
        
        for row, col in seats_to_book:
            movie.book_seat(row, col)
        return "Booking success."

    def available_movies(self, start_time, end_time):
        start_dt = datetime.strptime(start_time, '%H:%M')
        end_dt = datetime.strptime(end_time, '%H:%M')
        
        return [movie.name for movie in self.movies 
                if start_dt <= movie.start_time and movie.end_time <= end_dt]
