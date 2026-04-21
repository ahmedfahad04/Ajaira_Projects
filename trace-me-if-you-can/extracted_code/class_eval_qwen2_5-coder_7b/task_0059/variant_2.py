from datetime import datetime
import numpy as np

class Movie:
    def __init__(self, name, price, start_time, end_time, rows, cols):
        self.name = name
        self.price = price
        self.start_time = start_time
        self.end_time = end_time
        self.seats = np.zeros((rows, cols))

    def book_seat(self, row, col):
        if self.seats[row][col] == 0:
            self.seats[row][col] = 1
            return "Booked"
        else:
            return "Seat already booked"

class MovieBookingSystem:
    def __init__(self):
        self.movies = []

    def add_movie(self, name, price, start_time, end_time, rows, cols):
        self.movies.append(Movie(name, price, start_time, end_time, rows, cols))

    def book_ticket(self, movie_name, seats_to_book):
        for movie in self.movies:
            if movie.name == movie_name:
                results = [movie.book_seat(row, col) for row, col in seats_to_book]
                if "Seat already booked" in results:
                    return "Booking failed."
                return "Booking successful."
        return "Movie not found."

    def available_movies(self, start_time, end_time):
        available_movies = []
        for movie in self.movies:
            if start_time <= movie.start_time and movie.end_time <= end_time:
                available_movies.append(movie.name)
        return available_movies
