"""
bittensor-drand 1.3+ requires ``hotkey`` in ``get_encrypted_commit``; bittensor 9.7 omits it.

Import this module (side effect) before ``set_weights`` runs so commit-reveal works on localnet/testnet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from bittensor_drand import get_encrypted_commit
from numpy.typing import NDArray

from bittensor.core.extrinsics import commit_reveal as _cr
from bittensor.core.settings import version_as_int
from bittensor.utils.btlogging import logging
from bittensor.utils.weight_utils import convert_and_normalize_weights_and_uids

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor


def _commit_reveal_v3_extrinsic_with_hotkey(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    uids: Union[NDArray[np.int64], "torch.LongTensor", list],
    weights: Union[NDArray[np.float32], "torch.FloatTensor", list],
    version_key: int = version_as_int,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
    block_time: Union[int, float] = 12.0,
    period: Optional[int] = None,
) -> tuple[bool, str]:
    try:
        uids, weights = convert_and_normalize_weights_and_uids(uids, weights)

        current_block = subtensor.get_current_block()
        subnet_hyperparameters = subtensor.get_subnet_hyperparameters(
            netuid, block=current_block
        )
        tempo = subnet_hyperparameters.tempo
        subnet_reveal_period_epochs = subnet_hyperparameters.commit_reveal_period

        commit_for_reveal, reveal_round = get_encrypted_commit(
            uids=uids,
            weights=weights,
            version_key=version_key,
            tempo=tempo,
            current_block=current_block,
            netuid=netuid,
            subnet_reveal_period_epochs=subnet_reveal_period_epochs,
            block_time=block_time,
            hotkey=wallet.hotkey.public_key,
        )

        success, message = _cr._do_commit_reveal_v3(
            subtensor=subtensor,
            wallet=wallet,
            netuid=netuid,
            commit=commit_for_reveal,
            reveal_round=reveal_round,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
        )

        if success is not True:
            logging.error(message)
            return False, message

        logging.success(
            f"[green]Finalized![/green] Weights committed with reveal round [blue]{reveal_round}[/blue]."
        )
        return True, f"reveal_round:{reveal_round}"

    except Exception as e:
        logging.error(f":cross_mark: [red]Failed. Error:[/red] {e}")
        return False, str(e)


_cr.commit_reveal_v3_extrinsic = _commit_reveal_v3_extrinsic_with_hotkey

# ``bittensor.core.subtensor`` does ``from ...commit_reveal import commit_reveal_v3_extrinsic`` at import
# time; rebinding only on the commit_reveal module leaves that stale reference unless we patch here too.
import sys

_subtensor_mod = sys.modules.get("bittensor.core.subtensor")
if _subtensor_mod is not None:
    _subtensor_mod.commit_reveal_v3_extrinsic = _commit_reveal_v3_extrinsic_with_hotkey
