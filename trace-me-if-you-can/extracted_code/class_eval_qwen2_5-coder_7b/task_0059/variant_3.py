from datetime import datetime
import numpy as np

def create_movie(name, price, start_time, end_time, rows, cols):
    return {
        'name': name,
        'price': price,
        'start_time': datetime.strptime(start_time, '%H:%M'),
        'end_time': datetime.strptime(end_time, '%H:%M'),
        'seats': np.zeros((rows, cols))
    }

def book_ticket(movie, seats_to_book):
    for seat in seats_to_book:
        if movie['seats'][seat[0]][seat[1]] == 0:
            movie['seats'][seat[0]][seat[1]] = 1
        else:
            return "Booking failed."
    return "Booking successful."

def available_movies(movies, start_time, end_time):
    return [
        movie['name']
        for movie in movies
        if start_time <= movie['start_time'] and movie['end_time'] <= end_time
    ]

class MovieBookingSystem:
    def __init__(self):
        self.movies = []

    def add_movie(self, name, price, start_time, end_time, rows, cols):
        self.movies.append(create_movie(name, price, start_time, end_time, rows, cols))

    def book_ticket(self, movie_name, seats_to_book):
        movie = next((movie for movie in self.movies if movie['name'] == movie_name), None)
        if movie:
            return book_ticket(movie, seats_to_book)
        return "Movie not found."

    def available_movies(self, start_time, end_time):
        return available_movies(self.movies, start_time, end_time)
