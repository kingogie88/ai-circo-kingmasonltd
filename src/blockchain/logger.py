"""
Blockchain Logging Module for transparent record-keeping of recycling operations
"""

import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
import hashlib

from web3 import Web3
from eth_account import Account
from eth_typing import Address
import eth_utils

logger = logging.getLogger(__name__)

# Smart contract ABI (simplified version)
RECYCLING_ABI = [
    {
        "inputs": [
            {"name": "materialType", "type": "string"},
            {"name": "quantity", "type": "uint256"},
            {"name": "quality", "type": "uint8"},
            {"name": "timestamp", "type": "uint256"},
            {"name": "facilityId", "type": "string"},
            {"name": "energyUsed", "type": "uint256"},
            {"name": "carbonCredits", "type": "uint256"}
        ],
        "name": "logRecyclingBatch",
        "outputs": [{"name": "success", "type": "bool"}],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"name": "batchId", "type": "bytes32"}
        ],
        "name": "getBatchDetails",
        "outputs": [
            {"name": "materialType", "type": "string"},
            {"name": "quantity", "type": "uint256"},
            {"name": "quality", "type": "uint8"},
            {"name": "timestamp", "type": "uint256"},
            {"name": "facilityId", "type": "string"},
            {"name": "energyUsed", "type": "uint256"},
            {"name": "carbonCredits", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

class BlockchainLogger:
    def __init__(self, 
                 network_url: str,
                 contract_address: str,
                 private_key: Optional[str] = None):
        """
        Initialize blockchain logger.
        
        Args:
            network_url: URL of the Ethereum network (e.g. Polygon)
            contract_address: Address of the recycling smart contract
            private_key: Private key for signing transactions
        """
        self.w3 = Web3(Web3.HTTPProvider(network_url))
        self.contract_address = eth_utils.to_checksum_address(contract_address)
        self.contract = self.w3.eth.contract(
            address=self.contract_address,
            abi=RECYCLING_ABI
        )
        
        if private_key:
            self.account = Account.from_key(private_key)
        else:
            self.account = None
            
        logger.info(f"Initialized BlockchainLogger on network: {network_url}")

    def log_recycling_batch(self, batch_data: Dict) -> str:
        """
        Log recycling batch data to blockchain.
        
        Args:
            batch_data: Dictionary containing batch information
            
        Returns:
            Transaction hash
        """
        try:
            if not self.account:
                raise ValueError("No account configured for signing transactions")
                
            # Prepare transaction data
            batch_id = self._generate_batch_id(batch_data)
            
            # Convert data types for smart contract
            tx_data = {
                "materialType": batch_data["material_type"],
                "quantity": int(batch_data["quantity"] * 1000),  # Convert to grams
                "quality": int(batch_data["quality"] * 100),  # Convert to percentage
                "timestamp": int(datetime.now().timestamp()),
                "facilityId": batch_data["facility_id"],
                "energyUsed": int(batch_data["energy_used"] * 1000),  # Convert to Wh
                "carbonCredits": int(batch_data["carbon_credits"] * 100)  # Convert to cents
            }
            
            # Build transaction
            nonce = self.w3.eth.get_transaction_count(self.account.address)
            
            tx = self.contract.functions.logRecyclingBatch(
                tx_data["materialType"],
                tx_data["quantity"],
                tx_data["quality"],
                tx_data["timestamp"],
                tx_data["facilityId"],
                tx_data["energyUsed"],
                tx_data["carbonCredits"]
            ).build_transaction({
                'from': self.account.address,
                'gas': 2000000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': nonce,
            })
            
            # Sign and send transaction
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.account.key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for transaction receipt
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            if tx_receipt['status'] == 1:
                logger.info(f"Successfully logged batch {batch_id}")
                return self.w3.to_hex(tx_hash)
            else:
                raise Exception("Transaction failed")
                
        except Exception as e:
            logger.error(f"Failed to log batch: {e}")
            return ""

    def get_batch_details(self, batch_id: str) -> Dict:
        """
        Retrieve batch details from blockchain.
        
        Args:
            batch_id: Unique identifier for the batch
            
        Returns:
            Dictionary containing batch details
        """
        try:
            # Call smart contract
            result = self.contract.functions.getBatchDetails(
                eth_utils.to_bytes(hexstr=batch_id)
            ).call()
            
            # Convert result to dictionary
            return {
                "material_type": result[0],
                "quantity": result[1] / 1000,  # Convert from grams
                "quality": result[2] / 100,  # Convert from percentage
                "timestamp": datetime.fromtimestamp(result[3]).isoformat(),
                "facility_id": result[4],
                "energy_used": result[5] / 1000,  # Convert from Wh
                "carbon_credits": result[6] / 100,  # Convert from cents
                "batch_id": batch_id
            }
            
        except Exception as e:
            logger.error(f"Failed to get batch details: {e}")
            return {}

    def verify_batch(self, batch_id: str, original_data: Dict) -> bool:
        """
        Verify batch data against blockchain record.
        
        Args:
            batch_id: Batch identifier
            original_data: Original batch data to verify
            
        Returns:
            bool indicating if data matches
        """
        try:
            # Get blockchain data
            chain_data = self.get_batch_details(batch_id)
            if not chain_data:
                return False
                
            # Compare relevant fields
            matches = (
                chain_data["material_type"] == original_data["material_type"] and
                abs(chain_data["quantity"] - original_data["quantity"]) < 0.001 and
                abs(chain_data["quality"] - original_data["quality"]) < 0.01 and
                chain_data["facility_id"] == original_data["facility_id"]
            )
            
            return matches
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False

    def _generate_batch_id(self, batch_data: Dict) -> str:
        """Generate unique batch identifier."""
        # Create deterministic string from batch data
        data_string = (
            f"{batch_data['material_type']}"
            f"{batch_data['quantity']}"
            f"{batch_data['quality']}"
            f"{batch_data['facility_id']}"
            f"{datetime.now().isoformat()}"
        )
        
        # Generate SHA-256 hash
        return "0x" + hashlib.sha256(
            data_string.encode()
        ).hexdigest()

    def get_carbon_credits(self, facility_id: str, timeframe: str = "1d") -> Dict:
        """
        Calculate total carbon credits for a facility.
        
        Args:
            facility_id: Facility identifier
            timeframe: Time window for calculation
            
        Returns:
            Dictionary with carbon credit metrics
        """
        try:
            # This would typically query the smart contract
            # For now, return dummy data
            return {
                "total_credits": 156.7,
                "credit_value_usd": 3134.00,
                "recycled_tons": 45.5,
                "energy_saved_kwh": 12500,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get carbon credits: {e}")
            return {} 