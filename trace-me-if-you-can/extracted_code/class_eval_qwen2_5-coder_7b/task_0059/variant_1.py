from datetime import datetime
import numpy as np

class MovieTheater:
    def __init__(self):
        self.screenings = []

    def schedule_film(self, movie_name, ticket_price, show_start, show_end, seat_rows, seat_cols):
        film_info = {
            'name': movie_name,
            'ticket_price': ticket_price,
            'show_start': datetime.strptime(show_start, '%H:%M'),
            'show_end': datetime.strptime(show_end, '%H:%M'),
            'seating': np.zeros((seat_rows, seat_cols))
        }
        self.screenings.append(film_info)

    def reserve_seats(self, film_name, reserved_seats):
        for screening in self.screenings:
            if screening['name'] == film_name:
                for seat in reserved_seats:
                    if screening['seating'][seat[0]][seat[1]] == 0:
                        screening['seating'][seat[0]][seat[1]] = 1
                    else:
                        return "Seat booking failed."
                return "Seats booked successfully."
        return "Film not found."

    def find_available_films(self, viewing_start, viewing_end):
        viewing_start = datetime.strptime(viewing_start, '%H:%M')
        viewing_end = datetime.strptime(viewing_end, '%H:%M')

        accessible_films = []
        for screening in self.screenings:
            if viewing_start <= screening['show_start'] and screening['show_end'] <= viewing_end:
                accessible_films.append(screening['name'])

        return accessible_films
