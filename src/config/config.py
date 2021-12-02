import logging
import os

logger = logging.getLogger()
ROOT_DIR = os.path.abspath(os.curdir)
PREDICTOR_PATH = "../models/shape_predictor_68_face_landmarks.dat"
CASCADE_PATH = "../models/haarcascade_frontalface_default.xml"
INDICESFACE = 68
LABELS_U = [
    "0",
    "1",
    "2",
    "4",
    "5",
    "6",
    "9",
    "10",
    "1+2",
    "1+3",
    "1+4",
    "1+5",
    "1+6",
    "5+2+7",
    "1+9",
    "10+44",
    "2+3",
    "2+4",
    "2+5",
    "2+6",
    "2+7",
    "2+8",
    "61+73",
    "4+3+5+43",
    "4+3+5+44",
    "3+5",
    "41",
    "4+2",
    "43",
    "44",
    "4+5",
    "46",
    "4+9",
    "5+62",
    "5+6+4",
    "5+6",
    "5+70",
    "61",
    "62",
    "63",
    "64",
    "4+2+70+71",
    "70",
    "71",
    "72",
    "73",
    "2+7+70+71",
    "4+43+70+71",
    "1+2+43+70+71",
    "4+10+42",
    "4+10+44",
    "1+2+9+43",
    "1+2+70+71",
    "44+70+71",
    "1+2+5+70+71",
    "42+1+4",
    "62+64",
    "1+2+3",
    "1+2+4",
    "1+2+5",
    "1+2+6",
    "1+42",
    "1+4+6",
    "42+43",
    "4+43",
    "1+4+9",
    "1+5",
    "42+61",
    "4+2+62",
    "1+2+10",
    "1+2",
    "1+2",
    "1+2",
    "1+2",
    "1+2+6",
    "1+2",
    "1+2",
    "4+42+44",
    "1+2+41",
    "1+2+42",
    "1+2+43",
    "1+2+44",
    "1+2+46",
    "1+2+5",
    "4+42+62",
    "1+2+5",
    "1+2+61",
    "1+2+62",
    "1+2+63",
    "1+2+64",
    "1+2+5",
    "2+42",
    "5+70+71",
    "2+46",
    "1+2+70",
    "1+2+71",
    "1+2+73",
    "2+5+6",
    "1+2+41+61",
    "1+2+41+62",
    "43+61",
    "43+70",
    "4+72",
    "43+73",
    "42",
    "4+44",
    "2+3+42+70",
    "43",
    "2+8+61",
    "4+43+44",
    "1+3+43",
    "1+3+44",
    "1+2+42+70+71",
    "1+9+42",
    "1+9+43",
    "4+9+44",
    "1+3+62",
    "1+3+63",
    "4+43+73",
    "4+44+70+71",
    "1+2+42+63",
    "4+2+3+64",
    "4+43",
    "4+44+46+70+71",
    "1+9",
    "1+2+42",
    "1+2+43",
    "4",
    "4",
    "70+71",
    "4+44+46",
    "1+4+42",
    "1+4+43",
    "4",
    "4",
    "4",
    "1+9+61",
    "4",
    "4",
    "4+44+62",
    "9+44",
    "1+4+64",
    "4+41",
    "4+42",
    "4+43",
    "4+44",
    "4+44+73",
    "4+46",
    "4+61",
    "4+62",
    "4+64",
    "4+71",
    "4+44",
    "4+73",
    "4+70+71",
    "4+42",
    "4+43",
    "4+44",
    "74",
]
LABELS_L = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "10+25",
    "25+62",
    "22+25",
    "9",
    "10",
    "25+70",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "24",
    "20",
    "25",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "34",
    "61+72",
    "12+25",
    "12+20+25+26",
    "33",
    "34",
    "35",
    "15+72",
    "15+16+20+25",
    "13+16+25",
    "18+22+25",
    "16+23+25",
    "72",
    "61",
    "62",
    "12+22+25+26",
    "20+22+25+26",
    "15+16+17",
    "26+28",
    "21+17",
    "12+19+25",
    "15+16+20",
    "72",
    "73",
    "26+33",
    "15+16+25",
    "16+17",
    "16+20",
    "26",
    "16+23",
    "16+25",
    "16+26",
    "15+17+20+25+26",
    "28",
    "26+62",
    "22+23+25",
    "19+22+25+26",
    "10+25+26",
    "10+25+27",
    "22+25+26",
    "22+25+26+72",
    "22+25+26+73",
    "16+70",
    "25+72",
    "10+19+25+26",
    "25+27+70+71",
    "12+25+26",
    "12+25+27",
    "12+25+28",
    "15+22+25",
    "15+16+25+26",
    "15+17+20",
    "12+20+25",
    "15+17+22",
    "15+17+23",
    "10+25+72",
    "12+15+17",
    "12+25+41",
    "22+23",
    "15+17+24",
    "22+25",
    "22+26",
    "15+17+25",
    "15+17+26",
    "15+17+28",
    "17+20",
    "20+24+26",
    "17+23",
    "17+24",
    "17+25",
    "17+26",
    "12+15",
    "14+25+26",
    "12+17",
    "17+22+25",
    "18+70+71",
    "12+20",
    "12+22",
    "12+23",
    "12+24",
    "12+25",
    "12+26",
    "17+34",
    "12+28",
    "12+25+72",
    "12+25+73",
    "15+16+25+70",
    "15+16+25+72",
    "15+17+62",
    "25+26+28",
    "19+25",
    "12+28+72",
    "15+17+71",
    "16+25+25",
    "17+26",
    "16+25+26",
    "17+20+25",
    "22+25",
    "19+22+25",
    "15+20+25+26",
    "16+19+25+26",
    "32",
    "16+20+25",
    "34",
    "17+72",
    "12+16+25+72",
    "22+54",
    "12172526",
    "17+20+25+36",
    "10+16+25",
    "15+28+25",
    "12+17+20+26",
    "15+16+17+20+26",
    "18+25+26",
    "25+26+72",
    "25+26+73",
    "28+25",
    "16+25+72",
    "23+26+34",
    "15+16+19+25+26",
    "28+32",
    "10+25",
    "23+24",
    "23+25",
    "23+26",
    "12+16+25",
    "25+26+28+73",
    "18+22",
    "23+34",
    "20+25+26",
    "18+25",
    "13+14",
    "18+26",
    "20+25+27",
    "20+25+32",
    "16+20+25+26",
    "12+17+20+25+26",
    "18+34",
    "13+23",
    "15+22+25+72",
    "13+25",
    "18+22",
    "25+26",
    "13+28",
    "25+27",
    "20+15+17",
    "10+15+17+25",
    "10+16+19+25+26",
    "28+72",
    "28+73",
    "25+27+28",
    "23+70+71",
    "10+12+16+25+72",
    "22+25+26",
    "22+25+27",
    "16+22+25+26",
    "23+70",
    "23+71",
    "23+72",
    "23+73",
    "15+25",
    "19+25+26",
    "18+72",
    "18+73",
    "10+22+25",
    "13+19+25",
    "15+17+19+25",
    "24+25+26",
    "10+16+25+26",
    "17+20+25+26",
    "22+25+61",
    "17+20+25+27",
    "17+20+25+28",
    "25+27+72",
    "25+27+73",
    "17+20+25+29",
    "17+20+25+30",
    "17+20+25+31",
    "16+25",
    "10+12+25",
    "17+20+25+32",
    "22+25+72",
    "22+25+73",
    "17+20+25+33",
    "17+20+25+34",
    "17+20+25+35",
    "12+22+25",
    "13+72",
    "15+19+25",
    "12+17+20",
    "24+26",
    "24+28",
    "12+17+25",
    "15+16+17+25",
    "25+26+70+71",
    "19+22",
    "19+25",
    "34+62",
    "19+28",
    "15+16+25",
    "17+24+28",
    "10+16+25+72",
    "14+23",
    "34+72",
    "14+25",
    "34+73",
    "15+18+22+25",
    "19+25+26+28",
    "25+26",
    "10+12+16+25",
    "18+20+25+26",
    "26+20+25+26",
    "18",
    "24+70",
    "24+72",
    "20+15+17+20",
    "13+25+26",
    "15+17+25+26",
    "22+25+70+71",
    "16+22+25",
    "34",
    "18+22+25+70+71",
    "15+23+25+26",
    "15+17+20+25",
    "15+17+20+26",
    "16+17+25",
    "18+22+25+26",
    "34+70+71",
    "18+22+25+72",
    "19+25+26+72",
    "19+25+26",
    "19+25+27",
    "15+25+26",
    "15+19+25+26",
    "18+22+25",
    "18+22+26",
    "19+25+28",
    "25+16",
    "15+17+26",
    "72",
    "10+16",
    "15+20+25",
    "15+20+26",
    "18+22+25+73",
    "25",
    "25+26",
    "25+27",
    "25+28",
    "19+20+25+27",
    "20+15+17+71",
    "25+31",
    "25+32",
    "20+24",
    "20+25",
    "20+26",
    "20+27",
    "15+16",
    "15+17",
    "15+18",
    "17+25+26",
    "15+20",
    "23+24+25",
    "15+22",
    "15+23",
    "15+24",
    "15+25",
    "15+26",
    "10+15",
    "15+28",
    "10+17",
    "17+20+26",
    "20+25+26",
    "10+23",
]
