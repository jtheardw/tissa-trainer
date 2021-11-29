use std::fs::File;
use std::io::prelude::*;
use std::io::{BufReader, BufRead, BufWriter, Write};

const PAWN: i32 = 0;
const KNIGHT: i32 = 1;
const BISHOP: i32 = 2;
const ROOK: i32 = 3;
const QUEEN: i32 = 4;
const KING: i32 = 5;

const WHITE: i32 = 0;
const BLACK: i32 = 6;

fn input_number(piece: i32, white: bool, rank: i32, file: i32) -> i16 {
    let piece_num = if white { piece } else { 6 + piece };
    let idx = rank * 8 + file;
    let num = (piece_num * 64 + idx) as i16;
    if num > 769 {
        panic!("num too big {}, {} {} {} {}", num, piece, white, rank, file);
    }
    return num;
}

#[derive(Clone)]
pub struct Data {
    pub input: Vec<i16>,
    pub score: i16,
    pub outcome: i8
}

pub fn count_samples(fname: &str) -> std::io::Result<i64> {
    let mut total: i64 = 0;

    let mut file = File::open(fname)?;
    let mut r = BufReader::new(file);

    let mut buf = String::new();

    loop {
        let num_bytes = r.read_line(&mut buf)?;
        if num_bytes == 0 {break;}
        buf.clear();
        total += 1;
    }
    println!("Loading {} samples", total);

    return Ok(total);
}

fn fen_to_features(fen: String) -> Vec<i16> {
    let mut inputs: Vec<i16> = Vec::new();
    let mut rank: i32 = 7;
    let mut file: i32 = 0;

    let mut fen_split = fen.split(' ');
    let position = match fen_split.next() {
        Some(s) => String::from(s),
        None => panic!("bad FEN string")
    };

    for c in position.as_bytes().iter() {
        let c = *c;
        match c {
            b'K' => {inputs.push(input_number(KING, true, rank, file)); file += 1;},
            b'k' => {inputs.push(input_number(KING, false, rank, file)); file += 1;},

            b'Q' => {inputs.push(input_number(QUEEN, true, rank, file)); file += 1;},
            b'q' => {inputs.push(input_number(QUEEN, false, rank, file)); file += 1;},

            b'R' => {inputs.push(input_number(ROOK, true, rank, file)); file += 1;},
            b'r' => {inputs.push(input_number(ROOK, false, rank, file)); file += 1;},

            b'B' => {inputs.push(input_number(BISHOP, true, rank, file)); file += 1;},
            b'b' => {inputs.push(input_number(BISHOP, false, rank, file)); file += 1;},

            b'N' => {inputs.push(input_number(KNIGHT, true, rank, file)); file += 1;},
            b'n' => {inputs.push(input_number(KNIGHT, false, rank, file)); file += 1;},

            b'P' => {inputs.push(input_number(PAWN, true, rank, file)); file += 1;},
            b'p' => {inputs.push(input_number(PAWN, false, rank, file)); file += 1;},

            b'/' => {rank -= 1; file = 0;},
            b'1' => {file += 1},
            b'2' => {file += 2},
            b'3' => {file += 3},
            b'4' => {file += 4},
            b'5' => {file += 5},
            b'6' => {file += 6},
            b'7' => {file += 7},
            b'8' => {file += 8},
            _ => {}
        }
    }

    match fen_split.next() {
        Some(s) => {if s == "w" {inputs.push(768);}},
        None => panic!("bad FEN string")
    }

    return inputs;
}

fn make_sample(line: String) -> Data {
    // expected format:
    // <fen>;...;score:<score>;...;outcome:<outcome>

    // get inputs from the fen
    let fen_end_idx = line.find(";").unwrap();
    let fen = line.as_str()[..fen_end_idx].to_string();
    let inputs = fen_to_features(fen);

    // get the score from the engine that searched that position
    let score_start_idx = line.find("score:").unwrap() + 6;
    let score_end_idx = score_start_idx + line.as_str()[score_start_idx..].to_string().find(";").unwrap();
    let score = line.as_str()[score_start_idx..score_end_idx].parse().unwrap();

    // get the outcome of the game that produced that position
    let outcome_start_idx = line.find("outcome:").unwrap() + 8;
    let outcome_str = line.as_str()[outcome_start_idx..].trim();
    let outcome = match outcome_str {
        "0.0" => 0,
        "0.5" => 1,
        "1.0" => 2,
        _ => {panic!("bad outcome string!")}
    };

    return Data {
        input: inputs,
        score: score,
        outcome: outcome
    };
}

pub fn load_dataset(fname: &str) -> std::io::Result<Vec<Data>> {
    let mut data: Vec<Data> = Vec::new();
    let mut file = File::open(fname)?;
    let mut r = BufReader::new(file);
    let mut buf = String::new();

    loop {
        let num_bytes = r.read_line(&mut buf)?;
        if num_bytes == 0 {break;}

        data.push(make_sample(format!("{}", buf)));
        buf.clear();
    }

    return Ok(data);
}
