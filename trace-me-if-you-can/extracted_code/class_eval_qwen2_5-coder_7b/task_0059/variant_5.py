from datetime import datetime
import numpy as np
import asyncio

async def create_movie(name, price, start_time, end_time, rows, cols):
    return {
        'name': name,
        'price': price,
        'start_time': datetime.strptime(start_time, '%H:%M'),
        'end_time': datetime.strptime(end_time, '%H:%M'),
        'seats': np.zeros((rows, cols))
    }

async def book_ticket(movie, seats_to_book):
    for seat in seats_to_book:
        if movie['seats'][seat[0]][seat[1]] == 0:
            movie['seats'][seat[0]][seat[1]] = 1
        else:
            return "Booking failed."
    return "Booking successful."

async def available_movies(movies, start_time, end_time):
    return [
        movie['name']
        for movie in movies
        if start_time <= movie['start_time'] and movie['end_time'] <= end_time
    ]

class MovieBookingSystem:
    def __init__(self):
        self.movies = []

    async def add_movie(self, name, price, start_time, end_time, n):
        movie = await create_movie(name, price, start_time, end_time, n, n)
        self.movies.append(movie)

    async def book_ticket(self, name, seats_to_book):
        movie = next((movie for movie in self.movies if movie['name'] == name), None)
        if movie:
            return await book_ticket(movie, seats_to_book)
        return "Movie not found."

    async def available_movies(self, start_time, end_time):
        return await available_movies(self.movies, start_time, end_time)
