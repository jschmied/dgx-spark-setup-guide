package com.example.bank;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.math.BigDecimal;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class NeutralTransferTest {
    @Mock AccountRepository repo;
    @InjectMocks TransferService service;

    // Build an Account regardless of the model's chosen entity shape, so the
    // neutral suite tests production LOGIC, not whether the model happened to
    // expose a (String, BigDecimal) constructor. Tries common ctor signatures,
    // then falls back to no-arg + setters (mirrors Sonnet's reflection trick).
    private Account acc(String owner, String bal) {
        BigDecimal balance = new BigDecimal(bal);
        try {
            try { return Account.class.getConstructor(String.class, BigDecimal.class).newInstance(owner, balance); }
            catch (NoSuchMethodException ignore) {}
            try { return Account.class.getConstructor(Long.class, String.class, BigDecimal.class).newInstance(null, owner, balance); }
            catch (NoSuchMethodException ignore) {}
            try { return Account.class.getConstructor(String.class, BigDecimal.class, java.lang.Long.class).newInstance(owner, balance, null); }
            catch (NoSuchMethodException ignore) {}
            Account a = Account.class.getDeclaredConstructor().newInstance();
            Account.class.getMethod("setOwner", String.class).invoke(a, owner);
            Account.class.getMethod("setBalance", BigDecimal.class).invoke(a, balance);
            return a;
        } catch (Exception e) {
            throw new RuntimeException("neutral suite could not construct Account (unsupported entity shape): " + e, e);
        }
    }

    @Test void transfersFunds() {
        Account from = acc("Alice", "100.00"), to = acc("Bob", "50.00");
        when(repo.findById(1L)).thenReturn(Optional.of(from));
        when(repo.findById(2L)).thenReturn(Optional.of(to));
        service.transfer(1L, 2L, new BigDecimal("30.00"));
        assertEquals(0, from.getBalance().compareTo(new BigDecimal("70.00")));
        assertEquals(0, to.getBalance().compareTo(new BigDecimal("80.00")));
    }

    @Test void insufficientFundsThrowsAndKeepsBalances() {
        Account from = acc("Alice", "20.00"), to = acc("Bob", "50.00");
        when(repo.findById(1L)).thenReturn(Optional.of(from));
        when(repo.findById(2L)).thenReturn(Optional.of(to));
        assertThrows(InsufficientFundsException.class, () -> service.transfer(1L, 2L, new BigDecimal("30.00")));
        assertEquals(0, from.getBalance().compareTo(new BigDecimal("20.00")));
        assertEquals(0, to.getBalance().compareTo(new BigDecimal("50.00")));
    }

    @Test void unknownAccountThrows() {
        when(repo.findById(1L)).thenReturn(Optional.of(acc("Alice", "100.00")));
        when(repo.findById(2L)).thenReturn(Optional.empty());
        assertThrows(AccountNotFoundException.class, () -> service.transfer(1L, 2L, new BigDecimal("10.00")));
    }

    @Test void nonPositiveAmountThrows() {
        assertThrows(IllegalArgumentException.class, () -> service.transfer(1L, 2L, new BigDecimal("0.00")));
        assertThrows(IllegalArgumentException.class, () -> service.transfer(1L, 2L, new BigDecimal("-5.00")));
    }

    @Test void sameAccountThrows() {
        assertThrows(IllegalArgumentException.class, () -> service.transfer(1L, 1L, new BigDecimal("10.00")));
    }
}
