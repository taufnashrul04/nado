"""
Nado Linked Signer Setup
Generates a new linked signer keypair and links it to your subaccount.

Usage:
  python setup_linked_signer.py

Required env vars:
  MAIN_WALLET_PK - your main wallet private key (used ONCE to sign the link tx)
"""

import os
import sys
import json
from pathlib import Path
from eth_account import Account


def main():
    main_pk = os.environ.get("MAIN_WALLET_PK")
    if not main_pk:
        print("❌ ERROR: MAIN_WALLET_PK env var not set")
        print("   Run: MAIN_WALLET_PK=0xYOUR_MAIN_KEY python setup_linked_signer.py")
        sys.exit(1)

    if not main_pk.startswith("0x"):
        main_pk = "0x" + main_pk

    # Verify main wallet
    main_account = Account.from_key(main_pk)
    print(f"📍 Main wallet address: {main_account.address}")

    # Generate new linked signer
    print("\n🔧 Generating new linked signer keypair...")
    linked_signer = Account.create()

    print(f"\n🔑 Linked Signer Address:     {linked_signer.address}")
    print(f"🔒 Linked Signer Private Key: 0x{linked_signer.key.hex()}")

    # Save to file
    output_dir = Path("/root/nado-bot/secrets")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "linked_signer.json"

    data = {
        "main_wallet_address": main_account.address,
        "linked_signer_address": linked_signer.address,
        "linked_signer_private_key": "0x" + linked_signer.key.hex(),
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    os.chmod(output_file, 0o600)
    print(f"\n💾 Saved to: {output_file} (mode 600)")

    # Now submit the link tx
    print("\n📡 Submitting link_signer transaction to Nado...")
    print("   (signing with MAIN WALLET key)")

    try:
        from nado_protocol.client import create_nado_client, NadoClientMode
        from nado_protocol.engine_client.types.execute import LinkSignerParams
        from nado_protocol.utils.bytes32 import subaccount_to_hex
    except ImportError:
        print("\n⚠️  Nado SDK not installed yet.")
        print("   Run: pip install nado-protocol")
        print("   Then re-run this script.")
        print(f"\n✅ Linked signer key SAVED to {output_file}")
        print("   You can manually link it later via:")
        print(f"     - UI: nado.xyz Settings > Linked Signers > {linked_signer.address}")
        print(f"     - SDK: link_signer({linked_signer.address})")
        sys.exit(0)

    client = create_nado_client(NadoClientMode.MAINNET, main_pk)
    sender_hex = subaccount_to_hex(client.context.signer.address, "default")
    link_params = LinkSignerParams(
        sender=sender_hex, signer=linked_signer.address
    )
    result = client.context.engine_client.link_signer(link_params)

    print(f"\n✅ SUCCESS! Linked signer set.")
    print(f"   Tx: {result}")
    print(f"\n🎉 Bot dapat sign tx pakai key: 0x{linked_signer.key.hex()[:10]}...")
    print(f"   Main wallet kamu sekarang AMAN — tidak perlu disentuh lagi.")
    print(f"\n📝 Bot config akan pakai file: {output_file}")


if __name__ == "__main__":
    main()
