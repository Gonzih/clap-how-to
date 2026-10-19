use clap::{CommandFactory, Parser, Subcommand};
use clap_how_to::{HowToArgs, Index, command_documents};

#[derive(Parser)]
#[command(name = "example")]
struct Cli {
    #[command(subcommand)]
    command: Action,
}
#[derive(Subcommand)]
enum Action {
    /// Save a recoverable copy of all project data.
    Backup,
    /// Permanently remove a saved snapshot.
    Delete,
    /// Read the current application version.
    Version,
    /// Find commands by meaning, offline. Results are never executed.
    #[command(name = "how-to", visible_alias = "how")]
    HowTo(HowToArgs),
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse().command {
        Action::HowTo(args) => {
            clap_how_to::validate_query(&args.query)?;
            let index = Index::new(command_documents(&Cli::command()))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&index.search(&args.query, 5)?)?
            );
        }
        _ => println!("Example only: no action executed."),
    }
    Ok(())
}
