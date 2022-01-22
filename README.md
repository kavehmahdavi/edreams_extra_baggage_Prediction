
# eDreams Predicting Extra Baggage  

I am an enthusiast of travelling but I am always disappointed with the flight purchase process, 
I always felt it tedious and not clear. Some day looking into Internet I found a dateset related
with travel bookings. I was interested in extra baggage step, looking for if it’s predictable or not.

In order to improve the purchase process, I wanted to know if it’s possible to classify if in a 
booking the user will add extra baggage or not. In this way will be easier to improve the purchase
process offering directly an option to the client.

## Dataset description:

It contains 50,000 rows and 18 columns. Each row is related with a booking and each column with a feature.

```js
ID: (integer) Contains the booking id, that is unique.
TIMESTAMP: (date) The date when the trip was booked.
WEBSITE: (string) Website where the trip was purchased. It is composed of a prefix, first to letters the website and last two letters the country.
GDS: (integer) Number of flights bought through the Global Distribution System.
NO GDS: (integer) Number of flights bought through other channels.
DEPARTURE: (date) Departure date.
ARRIVAL: (date) Arrival date.
ADULTS: (integer) Number of adults.
CHILDREN: (integer) Number of children.
INFANTS: (integer) Number of infants.
TRAIN: (boolean) Whether the booking contains train tickets or not.
DISTANCE: (float) Distance traveled.
DEVICE: (string) Device used for the purchase.
HAUL TYPE (string): Whether the trip was “Domestic”, “Continental” or “Intercontinental”.
TRIP TYPE: (string) Trips can be either “One Way”, “Round Trip” or “Multi-Destination”.
PRODUCT: (string) Bookings can contain only travel (“Trip”) or the travel and hotel (“Dynpack”).
SMS: (boolean) Indicates if the customer has selected a confirmation by SMS.
EXTRA BAGGAGE: (boolean) Our target.
```