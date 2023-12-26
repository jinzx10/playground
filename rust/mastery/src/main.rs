extern crate rand;
use rand::Rng;

fn attempt(win_prob: f64, goal: u32, range: u32, num_trials: u32) -> (f64, f64) {
    let mut rng = rand::thread_rng();
    let mut num_games: Vec<u32> = vec![0; num_trials as usize];

    let mut mastery: i32;
    let mut hist: Vec<bool> = vec![false; range as usize];

    for i in 0..num_trials {
        mastery = 0;
        for item in &mut hist { *item = false; }

        while mastery < (goal as i32) {
            let idx: u32 = num_games[i as usize] % range;
            let success: bool = (rng.gen::<f64>()) < win_prob;
            mastery += (success as i32) - (hist[idx as usize] as i32);
            hist[idx as usize] = success;
            num_games[i as usize] += 1;
        }

    }

    // calculate mean & standard deviation
    let sum = num_games.iter().sum::<u32>();
    let mean: f64 = sum as f64 / num_trials as f64;
    let mut variance: f64 = 0.0;

    for i in 0..num_trials {
        variance += (num_games[i as usize] as f64 - mean).powi(2);
    }
    variance /= num_trials as f64;
    let std_dev: f64 = variance.sqrt();

    (mean, std_dev)
}

fn main() {
    let num_trials: u32 = 500000;
    let win_prob: f64 = 0.4104;
    let range: u32 = 100;

    let min_goal: u32 = 1;
    let max_goal: u32 = 55;
    let stride: u32 = 1;

    println!("Mastery/{}      Mean      StdDev    WinProb={:.2}%", range, win_prob*100.0);
    for goal in (min_goal..max_goal+1).step_by(stride as usize) {
        let (mean, std_dev) = attempt(win_prob, goal, range, num_trials);
        println!("{:6}       {:8.1}    {:8.1}", goal, mean, std_dev);
    }

}
