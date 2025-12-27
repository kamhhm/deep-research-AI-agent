#!/bin/bash
# Notify when current batch is complete and ready for parallel upload

cd "$(dirname "$0")"

echo "Monitoring current batch completion..."
echo "Will alert when ready to run parallel script"
echo ""

while true; do
    # Check batch status
    status=$(python helper_scripts/check_all_batches_status.py 2>/dev/null | grep "Batch 1:" | grep -o "COMPLETED" || echo "")
    
    if [ "$status" = "COMPLETED" ]; then
        echo ""
        echo "🎉 ============================================"
        echo "   BATCH 1 COMPLETE!"
        echo "   ============================================"
        echo ""
        echo "   Ready to run parallel upload for remaining batches!"
        echo ""
        echo "   Run this command:"
        echo "   cd \"$(pwd)\""
        echo "   python MTA_multi_batch_PARALLEL.py"
        echo ""
        
        # Mac notification
        osascript -e 'display notification "Batch 1 complete! Ready for parallel upload." with title "Week 10 Experiment"' 2>/dev/null
        
        # Play sound
        afplay /System/Library/Sounds/Glass.aiff 2>/dev/null
        
        break
    fi
    
    # Wait 5 minutes before checking again
    sleep 300
done

