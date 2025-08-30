#!/usr/bin/env python3
"""
Demonstration of TemplateFlow Error Handling

This script demonstrates ParcelExtract's improved error handling when
TemplateFlow atlases are not found, showing helpful error messages with
suggestions for available atlases and descriptions.

Key improvements demonstrated:
1. Clear error messages when atlas descriptions don't exist
2. Listing of available descriptions for existing atlases
3. Listing of available atlases for template spaces
4. Helpful suggestions instead of cryptic "No such file" errors
"""

from parcelextract.atlases.templateflow import TemplateFlowManager


def demonstrate_helpful_errors():
    """Demonstrate improved error handling with helpful messages."""
    print("="*70)
    print("TEMPLATEFLOW ERROR HANDLING DEMONSTRATION")
    print("="*70)
    
    tf_manager = TemplateFlowManager()
    
    # Test 1: Nonexistent description for existing atlas
    print("\n1. NONEXISTENT DESCRIPTION ERROR")
    print("-" * 40)
    print("Trying: Schaefer2018 with desc='512dimensions' (doesn't exist)")
    
    try:
        atlas_path = tf_manager.get_atlas('Schaefer2018', 'MNI152NLin2009cAsym', desc='512dimensions')
        print(f"Unexpected success: {atlas_path}")
    except Exception as e:
        print("Error message received:")
        print(str(e))
    
    # Test 2: Show available descriptions
    print("\n\n2. AVAILABLE DESCRIPTIONS LISTING")
    print("-" * 40)
    
    descriptions = tf_manager.list_available_descriptions('Schaefer2018', 'MNI152NLin2009cAsym')
    print(f"Available descriptions for Schaefer2018:")
    for i, desc in enumerate(descriptions, 1):
        print(f"  {i:2d}. {desc}")
    
    # Test 3: Show available atlases
    print("\n\n3. AVAILABLE ATLASES LISTING")
    print("-" * 35)
    
    atlases = tf_manager.list_available_atlases('MNI152NLin2009cAsym')
    print(f"Available atlases for MNI152NLin2009cAsym:")
    for i, atlas in enumerate(atlases, 1):
        print(f"  {i}. {atlas}")
    
    # Test 4: Nonexistent atlas entirely
    print("\n\n4. NONEXISTENT ATLAS ERROR")
    print("-" * 30)
    print("Trying: NonexistentAtlas2024 (doesn't exist)")
    
    try:
        atlas_path = tf_manager.get_atlas('NonexistentAtlas2024', 'MNI152NLin2009cAsym')
        print(f"Unexpected success: {atlas_path}")
    except Exception as e:
        print("Error message received:")
        print(str(e))
    
    # Test 5: Success case for comparison
    print("\n\n5. SUCCESSFUL DOWNLOAD (for comparison)")
    print("-" * 45)
    print("Trying: Schaefer2018 with desc='400Parcels7Networks' (should work)")
    
    try:
        atlas_path = tf_manager.get_atlas('Schaefer2018', 'MNI152NLin2009cAsym', desc='400Parcels7Networks')
        print(f"✅ Success! Atlas downloaded to:")
        print(f"   {atlas_path}")
        
        # Verify file exists
        import os
        if os.path.exists(atlas_path):
            size_mb = os.path.getsize(atlas_path) / (1024 * 1024)
            print(f"   File size: {size_mb:.2f} MB")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    print("\n" + "="*70)
    print("COMPARISON: OLD vs NEW ERROR MESSAGES")  
    print("="*70)
    print("\n🔴 OLD (cryptic): 'Error: No such file or no access: []'")
    print("\n🟢 NEW (helpful):")
    print("   Atlas 'Schaefer2018' with description '512dimensions' not found")
    print("   in TemplateFlow for template space 'MNI152NLin2009cAsym'.")
    print("")
    print("   Available descriptions for Schaefer2018:")
    print("   '100Parcels7Networks', '100Parcels17Networks', '200Parcels7Networks', ...")
    
    print("\n" + "="*70)
    print("BENEFITS OF IMPROVED ERROR HANDLING")
    print("="*70)
    print("✅ Clear identification of what went wrong")
    print("✅ Exact parameters that were attempted") 
    print("✅ List of valid alternatives to try")
    print("✅ No more cryptic '[]' error messages")
    print("✅ Faster debugging and user problem-solving")


def demonstrate_cli_usage():
    """Show how the improved errors appear in CLI usage."""
    print("\n" + "="*70)
    print("CLI USAGE EXAMPLES")
    print("="*70)
    
    print("\n🔴 This will now show a helpful error:")
    print("   uv run parcelextract --input data.nii.gz --atlas Schaefer2018 --desc 512dimensions")
    print("")
    print("🟢 Instead of the old cryptic error, users now see:")
    print("   Error: Atlas 'Schaefer2018' with description '512dimensions' not found...")
    print("   Available descriptions for Schaefer2018: '100Parcels7Networks', ...")
    
    print("\n🟢 Working examples:")
    print("   uv run parcelextract --input data.nii.gz --atlas Schaefer2018 --desc 400Parcels7Networks")
    print("   uv run parcelextract --input data.nii.gz --atlas Schaefer2018 --desc 800Parcels17Networks")


def main():
    """Run all demonstrations."""
    print("ParcelExtract: TemplateFlow Error Handling Improvements")
    print("=" * 60)
    print("\nThis demonstration shows how ParcelExtract now provides helpful")
    print("error messages when TemplateFlow atlases are not found, instead")
    print("of the previous cryptic 'No such file or no access: []' error.")
    
    try:
        demonstrate_helpful_errors()
        demonstrate_cli_usage()
        
    except ImportError:
        print("\n❌ TemplateFlow is not installed.")
        print("Install it with: uv add templateflow")
        return
    
    print(f"\n🎉 Demo completed successfully!")
    print("\nUsers will now receive clear, actionable error messages")
    print("when requesting non-existent TemplateFlow atlases or descriptions.")


if __name__ == '__main__':
    main()