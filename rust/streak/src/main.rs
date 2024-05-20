extern crate rand;
use rand::Rng;

fn attempt(win_prob: f64, goal: u32, num_trials: u32) -> (f64, f64) {
    let mut rng = rand::thread_rng();
    let mut num_games: Vec<u32> = vec![0; num_trials as usize];

    let mut streak: i32;

    for i in 0..num_trials {

        streak = 0;
        while streak < (goal as i32) {
            let success: bool = (rng.gen::<f64>()) < win_prob;
            if success {
                streak += 1;
            } else {
                streak = 0;
            }
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

fn quick(win_prob: f64, goal: u32) -> f64 {
     (win_prob.powi(- (goal as i32)) - 1.0)/(1.0 - win_prob)
}

fn main() {
    let num_trials: u32 = 100000;
    let win_prob: f64 = 0.9072;

    let min_goal: u32 = 90;
    let max_goal: u32 = 100;
    let stride: u32 = 1;

    println!("Streak      Mean      Mean(theo.) StdDev    WinProb={:.2}%", win_prob*100.0);
    for goal in (min_goal..max_goal+1).step_by(stride as usize) {
        //let (mean, std_dev) = attempt(win_prob, goal, num_trials);
        //println!("{:6}   {:8.1}   {:8.1}   {:8.1}", goal, mean, quick(win_prob, goal), std_dev);
        println!("{:6}   {:8.1}", goal, quick(win_prob, goal));
    }

}
