morse_code = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.', 'G': '--.', 'H': '....',
    'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---', 'P': '.--.',
    'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
    'Y': '-.--', 'Z': '--..',
    '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-', '5': '.....',
    '6': '-....', '7': '--...', '8': '---..', '9': '----.',
    '.': '.-.-.-', ',': '--..--', '?': '..--..', "'": '.----.', '!': '-.-.--', '/': '-..-.',
    '(': '-.--.', ')': '-.--.-', '&': '.-...', ':': '---...', ';': '-.-.-.', '=': '-...-',
    '+': '.-.-.', '-': '-....-', '_': '..--.-', '"': '.-..-.', '$': '...-..-', '@': '.--.-.'
}

stations = ("4U1UN", "VE8AT", "W6WX", "KH6RS", "ZL6B", "VK6RBP", "JA2IGY", "RR9O", "VR2B", "4S7B", "ZS6DN", "5Z4B", "4X6TU",
            "OH2B", "CS3B", "LU4AA", "OA4B", "YV5B")

morse2bits = {'.':'10', '-':"1110", 'S':'00'}

stations_bits = dict()


for station in stations:
    in_morse = ''.join([morse_code[letter]+"S" for letter in station]) # la S marca un espacio entre letras
    in_morse = in_morse[:-1] # quitas la S final

    in_bits = ''.join([morse2bits[simbol] for simbol in in_morse])
    stations_bits[station] = in_bits

#print(stations_bits)

### Estadística
count_ones = sum([station_bits.count('1') for station_bits in stations_bits.values()])
count_zeros = sum([station_bits.count('0') for station_bits in stations_bits.values()])
total_bits = count_zeros + count_ones

avg_len = sum([len(station_bits) for station_bits in stations_bits.values()])/len(stations)


print("% of ones: " + str(count_ones/total_bits))
print("% of zeros: " + str(count_zeros/total_bits))
print("Average length: " + str(avg_len))
print("Minimun length: " + str(min([len(station) for station in stations_bits.values()])))
print("Maximun length: " + str(max([len(station) for station in stations_bits.values()])))
