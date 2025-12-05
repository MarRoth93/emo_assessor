#!/bin/bash
#SBATCH --job-name=analyze_dimensions
#SBATCH --output=/home/rothermm/new_assessor/logs/analyze_dimensions_%j.out
#SBATCH --error=/home/rothermm/new_assessor/logs/analyze_dimensions_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Print job information
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=================================================="

# Load required modules
module load miniconda
source $CONDA_ROOT/bin/activate
conda activate mediaeval

# Navigate to project directory
cd /home/rothermm/new_assessor

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p results/dimension_analysis

# Print environment info
echo ""
echo "Environment Information:"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Set paths
RATINGS_FILE="./ratings/per_image_Slider_mean_sd_from_wide.csv"
OUTPUT_DIR="./results/dimension_analysis"

# Optional: Specify dimensions to analyze in detail
# Leave empty for automatic selection
FOCUS_DIMS=""  # e.g., "Effort Obstruction Fairness"

# Run dimension analysis
echo "=================================================="
echo "Analyzing Dimension Quality"
echo "=================================================="
echo "Ratings: $RATINGS_FILE"
echo "Output: $OUTPUT_DIR"
echo ""

if [ -z "$FOCUS_DIMS" ]; then
    python scripts/01_analyze_dimensions.py \
        --ratings_file "$RATINGS_FILE" \
        --output_dir "$OUTPUT_DIR"
else
    python scripts/01_analyze_dimensions.py \
        --ratings_file "$RATINGS_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --focus_dims $FOCUS_DIMS
fi

# Print completion info
echo ""
echo "=================================================="
echo "Dimension analysis complete!"
echo "=================================================="
echo "Results saved to: $OUTPUT_DIR"
echo "  - Statistics: dimension_statistics.csv"
echo "  - Correlations: high_correlations.csv"
echo "  - Quality issues: data_quality_issues.csv"
echo "  - Visualizations: *.png"
echo ""
echo "Job completed at: $(date)"
echo "=================================================="
